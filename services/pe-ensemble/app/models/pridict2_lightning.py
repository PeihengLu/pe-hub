"""PyTorch Lightning training path for PRIDICT2 outcome-distribution models."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl  # type: ignore[reportMissingImports]

from pe_common.training import (
    build_lr_scheduler,
    fit_lightning_module,
    LightningTrainerConfig,
)

from ..training.progress_log import log_training_best, make_epoch_logger

from pridict2.pridict.pridictv2.dataset import construct_load_dataloaders
from pridict2.pridict.pridictv2.loss import CELoss
from pridict2.pridict.pridictv2.model import (
    AnnotEmbeder_InitSeq,
    AnnotEmbeder_MutSeq,
    FeatureEmbAttention,
    MLPDecoderDistribution,
    MLPEmbedder,
    MaskGenerator,
    init_params_,
)
from pridict2.pridict.pridictv2.utilities import ReaderWriter, freeze_layers
from pridict2.pridict.rnn.rnn import RNN_Net


PRIDICT_BATCH = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    str,
]


def build_pridict_loss(loss_func: str) -> nn.Module:
    name = str(loss_func).strip()
    if name in {"MSEloss", "RMSEloss"}:
        return nn.MSELoss(reduction="mean")
    if name == "L1loss":
        return nn.L1Loss(reduction="mean")
    if name == "Huberloss":
        return nn.SmoothL1Loss(reduction="mean")
    if name == "KLDloss":
        return nn.KLDivLoss(reduction="none")
    if name == "CEloss":
        return CELoss(reduction="none")
    raise ValueError(f"Unsupported PRIDICT2 loss_func: {loss_func}")


def _compute_z_dim(embed_dim: int, annot_embed: int, assemb_opt: str) -> int:
    if assemb_opt == "stack":
        init_embed_dim = embed_dim + 3 * annot_embed
        mut_embed_dim = embed_dim + 2 * annot_embed
    else:
        init_embed_dim = embed_dim
        mut_embed_dim = embed_dim
    return int(np.min([init_embed_dim, mut_embed_dim]) // 2)


class PERNNDistributionModel(nn.Module):
    """Composite PE_RNN_distribution model matching the vendor training graph."""

    COMPONENT_NAMES: Tuple[str, ...] = (
        "init_annot_embed",
        "mut_annot_embed",
        "init_encoder",
        "mut_encoder",
        "local_featemb_init_attn",
        "local_featemb_mut_attn",
        "global_featemb_init_attn",
        "global_featemb_mut_attn",
        "seqlevel_featembeder",
        "decoder",
    )

    def __init__(
        self,
        *,
        embed_dim: int,
        num_hidden_layers: int,
        bidirection: bool,
        p_dropout: float,
        annot_embed: int,
        assemb_opt: str,
        seqlevel_featdim: int,
        num_outcomes: int,
        rnn_class: type = nn.GRU,
        nonlin_func: Optional[nn.Module] = None,
        fdtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.fdtype = fdtype
        self.mask_gen = MaskGenerator()
        nonlin = nonlin_func or nn.ReLU()

        self.init_annot_embed = AnnotEmbeder_InitSeq(
            embed_dim=embed_dim,
            annot_embed=annot_embed,
            assemb_opt=assemb_opt,
        )
        self.mut_annot_embed = AnnotEmbeder_MutSeq(
            embed_dim=embed_dim,
            annot_embed=annot_embed,
            assemb_opt=assemb_opt,
        )
        z_dim = _compute_z_dim(embed_dim, annot_embed, assemb_opt)
        if assemb_opt == "stack":
            init_embed_dim = embed_dim + 3 * annot_embed
            mut_embed_dim = embed_dim + 2 * annot_embed
        else:
            init_embed_dim = embed_dim
            mut_embed_dim = embed_dim

        self.init_encoder = RNN_Net(
            input_dim=init_embed_dim,
            hidden_dim=embed_dim,
            z_dim=z_dim,
            device=torch.device("cpu"),
            num_hiddenlayers=num_hidden_layers,
            bidirection=bidirection,
            rnn_pdropout=p_dropout,
            rnn_class=rnn_class,
            nonlinear_func=nonlin,
            fdtype=fdtype,
        )
        self.mut_encoder = RNN_Net(
            input_dim=mut_embed_dim,
            hidden_dim=embed_dim,
            z_dim=z_dim,
            device=torch.device("cpu"),
            num_hiddenlayers=num_hidden_layers,
            bidirection=bidirection,
            rnn_pdropout=p_dropout,
            rnn_class=rnn_class,
            nonlinear_func=nonlin,
            fdtype=fdtype,
        )
        self.local_featemb_init_attn = FeatureEmbAttention(z_dim)
        self.local_featemb_mut_attn = FeatureEmbAttention(z_dim)
        self.global_featemb_init_attn = FeatureEmbAttention(z_dim)
        self.global_featemb_mut_attn = FeatureEmbAttention(z_dim)
        self.seqlevel_featembeder = MLPEmbedder(
            inp_dim=seqlevel_featdim,
            embed_dim=z_dim,
            mlp_embed_factor=1,
            nonlin_func=nonlin,
            pdropout=p_dropout,
            num_encoder_units=1,
        )
        self.decoder = MLPDecoderDistribution(
            5 * z_dim,
            embed_dim=z_dim,
            outp_dim=num_outcomes,
            mlp_embed_factor=2,
            nonlin_func=nonlin,
            pdropout=p_dropout,
            num_encoder_units=1,
        )

        for name in self.COMPONENT_NAMES:
            init_params_(getattr(self, name))

    def iter_components(self) -> Sequence[Tuple[str, nn.Module]]:
        return tuple((name, getattr(self, name)) for name in self.COMPONENT_NAMES)

    def set_device(self, device: torch.device) -> None:
        self.init_encoder.device = device
        self.mut_encoder.device = device

    def forward_batch(
        self,
        batch: PRIDICT_BATCH,
        *,
        device: torch.device,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            X_init_nucl,
            X_init_proto,
            X_init_pbs,
            X_init_rt,
            X_mut_nucl,
            X_mut_pbs,
            X_mut_rt,
            x_init_len,
            x_mut_len,
            seqlevel_feat,
            y_val,
            _indx,
            _seq_id,
        ) = batch

        X_init_nucl = X_init_nucl.to(device)
        X_init_proto = X_init_proto.to(device)
        X_init_pbs = X_init_pbs.to(device)
        X_init_rt = X_init_rt.to(device)
        X_mut_nucl = X_mut_nucl.to(device)
        X_mut_pbs = X_mut_pbs.to(device)
        X_mut_rt = X_mut_rt.to(device)
        seqlevel_feat = seqlevel_feat.type(self.fdtype).to(device)
        y_batch = y_val.type(self.fdtype).to(device)

        with torch.set_grad_enabled(requires_grad):
            X_init_batch = self.init_annot_embed(X_init_nucl, X_init_proto, X_init_pbs, X_init_rt)
            X_mut_batch = self.mut_annot_embed(X_mut_nucl, X_mut_pbs, X_mut_rt)

            x_init_m = self.mask_gen.create_content_mask(
                (X_init_batch.shape[0], X_init_batch.shape[1]),
                x_init_len,
            )
            x_mut_m = self.mask_gen.create_content_mask(
                (X_mut_batch.shape[0], X_mut_batch.shape[1]),
                x_mut_len,
            )

            _, z_init = self.init_encoder.forward_complete(
                X_init_batch, x_init_len, requires_grad=requires_grad
            )
            _, z_mut = self.mut_encoder.forward_complete(
                X_mut_batch, x_mut_len, requires_grad=requires_grad
            )

            max_seg_len = z_init.shape[1]
            init_mask = x_init_m[:, :max_seg_len].to(device)
            s_init_global, _ = self.global_featemb_init_attn(z_init, mask=init_mask)
            s_init_local, _ = self.local_featemb_init_attn(
                z_init, mask=X_init_rt[:, :max_seg_len]
            )

            max_seg_len = z_mut.shape[1]
            mut_mask = x_mut_m[:, :max_seg_len].to(device)
            s_mut_global, _ = self.global_featemb_mut_attn(z_mut, mask=mut_mask)
            s_mut_local, _ = self.local_featemb_mut_attn(
                z_mut, mask=X_mut_rt[:, :max_seg_len]
            )

            seqfeat = self.seqlevel_featembeder(seqlevel_feat)
            logits = self.decoder(
                torch.cat(
                    [s_init_global, s_init_local, s_mut_global, s_mut_local, seqfeat],
                    axis=-1,
                )
            )
        return logits, y_batch

    def load_vendor_statedict(self, statedict_dir: str, *, device: torch.device) -> None:
        for name, module in self.iter_components():
            path = os.path.join(statedict_dir, f"{name}.pkl")
            if not os.path.isfile(path):
                continue
            module.load_state_dict(
                torch.load(path, map_location=device, weights_only=False)
            )

    def save_vendor_statedict(self, statedict_dir: str) -> None:
        os.makedirs(statedict_dir, exist_ok=True)
        for name, module in self.iter_components():
            torch.save(module.state_dict(), os.path.join(statedict_dir, f"{name}.pkl"))


class _PRIDICT2LightningModule(pl.LightningModule):
    def __init__(
        self,
        model: PERNNDistributionModel,
        *,
        train_hparams: Mapping[str, Any],
        loss_func_name: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = model
        self.hparams_map = dict(train_hparams)
        self.loss_func_name = str(loss_func_name)
        self.loss_fn = build_pridict_loss(self.loss_func_name)
        self.device_ref = device

    def _step(self, batch: PRIDICT_BATCH, *, train: bool) -> torch.Tensor:
        logits, target = self.model.forward_batch(
            batch,
            device=self.device_ref,
            requires_grad=train,
        )
        loss = self.loss_fn(logits, target)
        if self.loss_func_name == "KLDloss":
            loss = loss.sum(dim=-1).mean()
        elif self.loss_func_name in {"CEloss"}:
            loss = loss.sum(dim=-1).mean()
        return loss

    def training_step(self, batch: PRIDICT_BATCH, _batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, train=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: PRIDICT_BATCH, _batch_idx: int) -> torch.Tensor:
        loss = self._step(batch, train=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=float(self.hparams_map.get("lr", 1e-4)),
            weight_decay=float(self.hparams_map.get("weight_decay", 1e-4)),
        )
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_name=str(self.hparams_map.get("scheduler", "none")),
            scheduler_kwargs=cast(
                Optional[Dict[str, Any]],
                self.hparams_map.get("scheduler_kwargs"),
            ),
        )
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


def build_pernn_distribution_model(
    hyperparameters: Mapping[str, Any],
    *,
    seqlevel_featdim: int,
    num_outcomes: int,
    device: torch.device,
) -> PERNNDistributionModel:
    from torch import nn as torch_nn

    embed_dim = int(hyperparameters.get("embed_dim", 64))
    num_hidden_layers = int(hyperparameters.get("num_hidden_layers", 1))
    bidirection = bool(hyperparameters.get("bidirection", True))
    p_dropout = float(
        hyperparameters.get(
            "p_dropout",
            hyperparameters.get("dropout", 0.1),
        )
    )
    annot_embed = int(hyperparameters.get("annot_embed", 8))
    assemb_opt = str(hyperparameters.get("assemb_opt", "add"))
    model = PERNNDistributionModel(
        embed_dim=embed_dim,
        num_hidden_layers=num_hidden_layers,
        bidirection=bidirection,
        p_dropout=p_dropout,
        annot_embed=annot_embed,
        assemb_opt=assemb_opt,
        seqlevel_featdim=seqlevel_featdim,
        num_outcomes=num_outcomes,
        rnn_class=torch_nn.GRU,
        nonlin_func=torch_nn.ReLU(),
    )
    model.set_device(device)
    model.to(device)
    return model


def apply_pridict_freezing(
    model: PERNNDistributionModel,
    *,
    trainable_layernames: Sequence[str],
) -> None:
    freeze_layers(model.iter_components(), list(trainable_layernames))


def build_pridict_dataloaders(
    *,
    train_dataset: Any,
    val_dataset: Any,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    config = {"batch_size": int(batch_size), "num_workers": 0}
    partition = {"train": train_dataset, "validation": val_dataset}
    loaders, _, _, _ = construct_load_dataloaders(
        partition,
        ["train", "validation"],
        config,
        wrk_dir=None,
    )
    return loaders["train"], loaders["validation"]


def save_pridict_run_artifacts(
    *,
    model_dir: Path,
    model: PERNNDistributionModel,
    config_map: Tuple[Any, Dict[str, Any]],
    best_epoch: int,
) -> None:
    model_dir = model_dir.expanduser().resolve()
    statedict_dir = model_dir / "model_statedict"
    config_dir = model_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    model.save_vendor_statedict(str(statedict_dir))
    mconfig, options = config_map
    ReaderWriter.dump_data(mconfig, str(config_dir / "mconfig.pkl"))
    ReaderWriter.dump_data(options, str(config_dir / "exp_options.pkl"))
    with open(statedict_dir / "best_epoch.pkl", "wb") as handle:
        pickle.dump({"epoch": int(best_epoch) + 1}, handle)


def train_pridict2_with_lightning(
    *,
    model: PERNNDistributionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    hyperparameters: Mapping[str, Any],
    device: torch.device,
    loss_func: str,
    progress_log: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    if bool(hyperparameters.get("freezing", False)):
        apply_pridict_freezing(
            model,
            trainable_layernames=list(
                hyperparameters.get("trainable_layernames", ["decoder"])
            ),
        )

    lightning_module = _PRIDICT2LightningModule(
        model,
        train_hparams=hyperparameters,
        loss_func_name=loss_func,
        device=device,
    )
    metrics = fit_lightning_module(
        lightning_module,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=LightningTrainerConfig(
            max_epochs=int(hyperparameters.get("num_epochs", 20)),
            grad_clip=float(hyperparameters.get("grad_clip", 1.0)),
            patience=int(hyperparameters.get("early_stopping_patience", 10)),
            min_delta=float(hyperparameters.get("early_stopping_min_delta", 0.0)),
            enable_progress_bar=bool(hyperparameters.get("progress_bar", False)),
            log_every_n_steps=int(hyperparameters.get("log_every_n_steps", 25)),
        ),
        on_epoch_end=make_epoch_logger(progress_log, cancel_check=cancel_check),
    )
    log_training_best(
        progress_log,
        best_epoch=int(metrics["best_epoch"]),
        best_val_loss=float(metrics["best_val_loss"]),
    )
    return metrics


def vendor_state_dict_path(statedict_root: Optional[str], run_num: int = 0) -> Optional[str]:
    if not statedict_root:
        return None
    path = os.path.join(
        statedict_root,
        "train_val",
        f"run_{run_num}",
        "model_statedict",
    )
    return path if os.path.isdir(path) else statedict_root
