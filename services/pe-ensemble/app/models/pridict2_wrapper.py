from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning.pytorch as pl  # type: ignore[reportMissingImports]

from .vendor_path import resolve_vendor_models_path
from . import weights_registry
from pe_common.model_interface import BasePEModel
from pe_common.training import (
    build_lr_scheduler,
    fit_lightning_module,
    LightningTrainerConfig,
    regression_metrics,
)
from pe_common.splits import (
    has_assigned_cv_folds,
    iter_assigned_cv_folds,
    resolve_train_val_from_splits,
)

from ..training.progress_log import log_training_best, make_epoch_logger

# Add vendor model paths required by PRIDICT2 imports.
_vendor_root = resolve_vendor_models_path()
_pridict2_root = resolve_vendor_models_path("pridict2")
if str(_vendor_root) not in sys.path:
    sys.path.insert(0, str(_vendor_root))
if str(_pridict2_root) not in sys.path:
    sys.path.insert(0, str(_pridict2_root))

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


# Vendor notebooks fix annot_embed=8 and assemb_opt='stack'. Neither is a
# tunable hyperparameter: 'add' requires matching embed widths; z_dim is
# derived from embed_dim + annot_embed under stack assembly.
_ANNOT_EMBED = 8
_ASSEMB_OPT = "stack"


def _compute_z_dim(embed_dim: int, annot_embed: int = _ANNOT_EMBED) -> int:
    init_embed_dim = embed_dim + 3 * annot_embed
    mut_embed_dim = embed_dim + 2 * annot_embed
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
        seqlevel_featdim: int,
        num_outcomes: int,
        rnn_class: type = nn.GRU,
        nonlin_func: Optional[nn.Module] = None,
        fdtype: torch.dtype = torch.float32,
        annot_embed: int = _ANNOT_EMBED,
    ) -> None:
        super().__init__()
        self.fdtype = fdtype
        self.mask_gen = MaskGenerator()
        nonlin = nonlin_func or nn.ReLU()

        self.init_annot_embed = AnnotEmbeder_InitSeq(
            embed_dim=embed_dim,
            annot_embed=annot_embed,
            assemb_opt=_ASSEMB_OPT,
        )
        self.mut_annot_embed = AnnotEmbeder_MutSeq(
            embed_dim=embed_dim,
            annot_embed=annot_embed,
            assemb_opt=_ASSEMB_OPT,
        )
        z_dim = _compute_z_dim(embed_dim, annot_embed)
        init_embed_dim = embed_dim + 3 * annot_embed
        mut_embed_dim = embed_dim + 2 * annot_embed

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
    ) -> None:
        super().__init__()
        self.model = model
        self.hparams_map = dict(train_hparams)
        self.loss_func_name = str(loss_func_name)
        self.loss_fn = build_pridict_loss(self.loss_func_name)

    def _step(self, batch: PRIDICT_BATCH, *, train: bool) -> torch.Tensor:
        # Use Lightning's live device (not a stale constructor capture). After
        # fit_end Lightning parks the module on CPU; callers must ``.to(device)``.
        logits, target = self.model.forward_batch(
            batch,
            device=self.device,
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
    embed_dim = int(hyperparameters.get("embed_dim", 64))
    num_hidden_layers = int(hyperparameters.get("num_hidden_layers", 1))
    bidirection = bool(hyperparameters.get("bidirection", True))
    p_dropout = float(
        hyperparameters.get(
            "p_dropout",
            hyperparameters.get("dropout", 0.1),
        )
    )
    annot_embed = _ANNOT_EMBED
    model = PERNNDistributionModel(
        embed_dim=embed_dim,
        num_hidden_layers=num_hidden_layers,
        bidirection=bidirection,
        p_dropout=p_dropout,
        annot_embed=annot_embed,
        seqlevel_featdim=seqlevel_featdim,
        num_outcomes=num_outcomes,
        rnn_class=nn.GRU,
        nonlin_func=nn.ReLU(),
    )
    model.set_device(device)
    model.to(device)
    return model


def apply_pridict_freezing(
    model: PERNNDistributionModel,
    *,
    trainable_layernames: Sequence[str],
) -> None:
    # Vendor freeze_layers expects ``(module, name)`` pairs.
    freeze_layers(
        [(module, name) for name, module in model.iter_components()],
        list(trainable_layernames),
    )


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
    # Encoder modules keep an explicit ``.device`` attribute used for masks.
    model.set_device(device)
    model.to(device)
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


class PRIDICT2ModelWrapper(BasePEModel):
    """Wrapper for PRIDICT2 model (outcome distribution prediction)"""

    # Native PRIDICT/PRIDICT2 input columns (output of PE-DB's pridict converter).
    PRIDICT_REQUIRED_COLUMNS = {
        "seq_id",
        "wide_initial_target",
        "wide_mutated_target",
        "deepeditposition",
        "deepeditposition_lst",
        "Correction_Type",
        "Correction_Length",
        "protospacerlocation_only_initial",
        "PBSlocation",
        "RT_initial_location",
        "RT_mutated_location",
    }
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        wsize: int = 20,
        model_name: Optional[str] = None,
    ):
        """
        Initialize PRIDICT2 model wrapper.

        Args:
            device: PyTorch device
            wsize: Window size for sequence processing
            model_name: Optional legacy vendor base-model name used only when
                preparing data without loaded weights (prefer explicit weights).
        """
        super().__init__('PRIDICT2', device)
        self.wsize = wsize
        self.model_name_str = model_name

        from pridict2.pridict.pridictv2.predict_outcomedistrib import PRIEML_Model
        
        self.prieml_model = PRIEML_Model(
            device=device or torch.device('cpu'),
            wsize=wsize,
            normalize='max',
            fdtype=torch.float32
        )
        self.model_components = None
        self.loaded_model_dir: Optional[str] = None
        self.selected_cell_type: Optional[str] = None

    def _default_outcomes(self) -> List[str]:
        return ["averageedited"]

    @staticmethod
    def _normalize_cell_type(name: str) -> str:
        return "".join(str(name).split("_"))

    @staticmethod
    def _cell_types_from_run_dir(model_path: str) -> List[str]:
        """Return prediction heads available on disk for a run directory."""
        state_dict_dir = Path(model_path) / "model_statedict"
        return sorted(
            PRIDICT2ModelWrapper._normalize_cell_type(path.stem.replace("decoder_", ""))
            for path in state_dict_dir.glob("decoder_*.pkl")
        )

    def _cell_types_from_loaded_config(self) -> List[str]:
        """Return prediction-head cell types from the loaded weight run config."""
        if not self.loaded_model_dir:
            raise ValueError("PRIDICT2 weights are not loaded.")
        import os

        mconfig_dir = os.path.join(self.loaded_model_dir, "config")
        _, options = self.prieml_model._load_model_config(mconfig_dir)
        datasets = options.get("datasets_name") or []
        return [self._normalize_cell_type(name) for name in datasets]

    @staticmethod
    def _registered_base_weight_ids() -> List[str]:
        return weights_registry.list_weight_ids("pridict2")

    @staticmethod
    def _split_weight_name(name: str) -> tuple[str, Optional[str]]:
        """Split ``{base_run_id}`` or ``{base_run_id}__{cell_type}``."""
        candidate = name.strip()
        if not candidate:
            raise ValueError("PRIDICT2 weight name is empty.")

        base_ids = PRIDICT2ModelWrapper._registered_base_weight_ids()
        for base_id in sorted(base_ids, key=len, reverse=True):
            if candidate == base_id:
                return base_id, None
            prefix = f"{base_id}__"
            if candidate.startswith(prefix):
                cell_type = candidate[len(prefix):]
                if cell_type:
                    return base_id, cell_type
                break

        if Path(candidate).expanduser().is_dir():
            return candidate, None

        raise ValueError(
            f"Unknown PRIDICT2 weights '{name}'. "
            f"Available: {PRIDICT2ModelWrapper.list_available_weights()}"
        )

    @staticmethod
    def resolve_weight_selection(name: str) -> tuple[Path, Optional[str]]:
        """Resolve a weight selection to a run directory and optional cell-type head."""
        candidate = name.strip()
        cell_type: Optional[str] = None
        base_ref = candidate

        if "__" in candidate:
            maybe_base, maybe_cell = PRIDICT2ModelWrapper._split_weight_name(candidate)
            base_ref = maybe_base
            cell_type = maybe_cell

        base_path = Path(base_ref).expanduser()
        if base_path.is_dir() and (base_path / "model_statedict").is_dir():
            run_dir = base_path.resolve()
        else:
            registry_id = base_ref.replace("/", "__")
            try:
                run_dir = weights_registry.resolve_dir("pridict2", registry_id)
            except ValueError:
                trained_root = resolve_vendor_models_path("pridict2", "trained_models")
                parts = base_ref.replace("\\", "/").split("/")
                if len(parts) == 3:
                    legacy = trained_root / parts[0] / parts[1] / "train_val" / parts[2]
                elif "__" in registry_id:
                    legacy_parts = registry_id.split("__")
                    if len(legacy_parts) == 3:
                        legacy = (
                            trained_root
                            / legacy_parts[0]
                            / legacy_parts[1]
                            / "train_val"
                            / legacy_parts[2]
                        )
                    else:
                        legacy = trained_root / registry_id
                else:
                    legacy = trained_root / registry_id
                if (legacy / "model_statedict").is_dir():
                    run_dir = legacy.resolve()
                else:
                    raise ValueError(
                        f"Unknown PRIDICT2 weights '{name}'. "
                        f"Available: {PRIDICT2ModelWrapper.list_available_weights()}"
                    ) from None

        PRIDICT2ModelWrapper._validate_run_dir(str(run_dir))
        available = PRIDICT2ModelWrapper._cell_types_from_run_dir(str(run_dir))
        if not available:
            # Ensemble-trained PE_RNN_distribution runs save a generic decoder.pkl
            # (no decoder_<cell> heads). Those are loadable as a single-head bundle.
            if PRIDICT2ModelWrapper._is_single_head_distribution_run(str(run_dir)):
                if cell_type is not None:
                    raise ValueError(
                        f"PRIDICT2 run '{base_ref}' is a single-head ensemble artifact "
                        "(decoder.pkl) with no cell-type heads. Use the base weights "
                        "id without a '__{cell}' suffix."
                    )
                return run_dir, None
            raise ValueError(f"PRIDICT2 run has no decoder heads: {run_dir}")

        if cell_type is None:
            if len(available) == 1:
                return run_dir, available[0]
            raise ValueError(
                "PRIDICT2 weight selection must include a cell-type head suffix "
                f"for multi-head runs. Choose one of: "
                f"{[f'{base_ref}__{head}' for head in available]}"
            )

        normalized = PRIDICT2ModelWrapper._normalize_cell_type(cell_type)
        if normalized not in available:
            raise ValueError(
                f"PRIDICT2 head '{cell_type}' is not available for '{base_ref}'. "
                f"Available heads: {available}"
            )
        return run_dir, normalized

    def _to_pridict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that ``df`` is already in PRIDICT's native schema.

        Standardized -> PRIDICT conversion is owned by the PE-DB service; fetch
        model-format data from ``GET /api/filter?...&format=pridict2`` rather than
        passing standardized rows here.
        """
        if self.PRIDICT_REQUIRED_COLUMNS.issubset(df.columns):
            out = df.copy()
            if "Correction_Length" in out.columns:
                out["Correction_Length"] = pd.to_numeric(
                    out["Correction_Length"], errors="raise"
                ).astype(int)
            return out
        missing = sorted(self.PRIDICT_REQUIRED_COLUMNS.difference(df.columns))
        raise ValueError(
            "PRIDICT2 expects native input columns; missing: "
            f"{missing}. Fetch model-format data from PE-DB "
            "(GET /api/filter?...&format=pridict2)."
        )

    def _predict_from_loaded_or_current_model(
        self, dloader: Any, y_ref: List[str]
    ) -> pd.DataFrame:
        if isinstance(self.model, PERNNDistributionModel):
            raise ValueError(
                "PERNNDistributionModel predictions require _predict_pernn_dataframe "
                "(dataframe input), not a vendor ConcatDataLoader."
            )
        if self.selected_cell_type and self.loaded_model_dir:
            return self.prieml_model.predict_from_dloader(
                dloader=dloader,
                model_dir=self.loaded_model_dir,
                y_ref=y_ref,
            )
        if self.model_components is not None:
            # PRIDICT2 wrapper exposes a "loaded-models" prediction entrypoint.
            return self.prieml_model.predict_from_dloader_using_loaded_models(
                dloader=dloader,
                models=self.model_components,
                y_ref=y_ref,
            )
        if self.loaded_model_dir:
            return self.prieml_model.predict_from_dloader(
                dloader=dloader,
                model_dir=self.loaded_model_dir,
                y_ref=y_ref,
            )
        raise ValueError("Model not loaded. Call load_model() first.")
    
    @staticmethod
    def _dataset_component_names(
        datasets_name: List[str],
        *,
        separate_attention_layers: bool,
        separate_seqlevel_embedder: bool,
    ) -> List[str]:
        """Return dataset-specific statedict filenames expected for a run config."""
        names: List[str] = []
        for raw_name in datasets_name:
            dname = "".join(str(raw_name).split("_"))
            names.append(f"decoder_{dname}.pkl")
            if separate_seqlevel_embedder:
                names.append(f"seqlevel_featembeder_{dname}.pkl")
            if separate_attention_layers:
                for seq_type in ("init", "mut"):
                    for attn_type in ("local", "global"):
                        names.append(f"{attn_type}_featemb_{seq_type}_attn_{dname}.pkl")
        return names

    @staticmethod
    def _validate_run_dir(model_path: str) -> None:
        """Ensure run config datasets match on-disk statedict component names."""
        import os
        import pickle

        config_dir = os.path.join(model_path, "config")
        state_dict_dir = os.path.join(model_path, "model_statedict")
        if not os.path.isdir(config_dir) or not os.path.isdir(state_dict_dir):
            raise ValueError(f"Invalid PRIDICT2 run directory: {model_path}")

        exp_options_path = os.path.join(config_dir, "exp_options.pkl")
        if not os.path.isfile(exp_options_path):
            raise ValueError(f"PRIDICT2 run config missing exp_options.pkl: {model_path}")

        with open(exp_options_path, "rb") as handle:
            options = pickle.load(handle)

        datasets = options.get("datasets_name") or []
        required = PRIDICT2ModelWrapper._dataset_component_names(
            datasets,
            separate_attention_layers=bool(options.get("separate_attention_layers")),
            separate_seqlevel_embedder=bool(options.get("separate_seqlevel_embedder")),
        )
        missing = [
            name
            for name in required
            if not os.path.isfile(os.path.join(state_dict_dir, name))
        ]
        if not missing:
            return

        on_disk = sorted(
            path.name
            for path in Path(state_dict_dir).glob("*.pkl")
            if path.name != "best_epoch.pkl"
            and any(
                token in path.name
                for token in ("decoder_", "seqlevel_featembeder_", "_featemb_")
            )
        )
        raise ValueError(
            "PRIDICT2 weight bundle is incomplete: "
            f"config expects datasets {datasets} but statedict is missing "
            f"{missing}. On-disk dataset-specific files: {on_disk}. "
            "This usually means the wrong model_statedict was packaged with the "
            "run config. Re-migrate from vendor or choose a compatible weight set."
        )

    @staticmethod
    def list_available_weights() -> List[str]:
        """List registered PRIDICT2 weight IDs, with cell-type head suffix when needed."""
        return [entry["id"] for entry in PRIDICT2ModelWrapper.list_available_weight_entries()]

    @staticmethod
    def list_available_weight_entries() -> List[Dict[str, Any]]:
        """List loadable PRIDICT2 weight entries for API/UI selection."""
        registry_by_id = {
            entry["id"]: entry for entry in weights_registry.list_entries("pridict2")
        }
        entries: List[Dict[str, Any]] = []
        for base_id in PRIDICT2ModelWrapper._registered_base_weight_ids():
            run_dir = weights_registry.resolve_dir("pridict2", base_id)
            try:
                PRIDICT2ModelWrapper._validate_run_dir(str(run_dir))
            except ValueError:
                continue

            manifest = dict(registry_by_id.get(base_id, {}))
            base_label = manifest.get("label", base_id.replace("__", " / "))
            cell_types = PRIDICT2ModelWrapper._cell_types_from_run_dir(str(run_dir))
            if not cell_types:
                if PRIDICT2ModelWrapper._is_single_head_distribution_run(str(run_dir)):
                    entries.append(
                        {
                            **manifest,
                            "id": base_id,
                            "model": "pridict2",
                            "label": base_label,
                            "cell_type": None,
                        }
                    )
                continue
            if len(cell_types) == 1:
                entries.append(
                    {
                        **manifest,
                        "id": base_id,
                        "model": "pridict2",
                        "label": base_label,
                        "cell_type": cell_types[0],
                    }
                )
                continue

            for cell_type in cell_types:
                weight_id = f"{base_id}__{cell_type}"
                entries.append(
                    {
                        **manifest,
                        "id": weight_id,
                        "model": "pridict2",
                        "label": f"{base_label} / {cell_type}",
                        "cell_type": cell_type,
                    }
                )
        return entries

    def _resolve_weights_dir(self, name: str) -> Path:
        """Resolve a weight set ID (or directory path) to a run directory."""
        run_dir, cell_type = self.resolve_weight_selection(name)
        self.selected_cell_type = cell_type
        return run_dir

    def load_weights_by_name(self, name: str) -> None:
        """Load a named pre-trained PRIDICT2 weight set.

        Args:
            name: A weight set name from :meth:`list_available_weights`, or a
                path to a trained run directory.
        """
        self.load_model(str(self._resolve_weights_dir(name)))

    def load_model(self, model_path: str) -> None:
        """
        Load a trained PRIDICT2 run directory.

        Ensemble-trained single-head ``PE_RNN_distribution`` artifacts (generic
        ``decoder.pkl`` names) are loaded into :class:`PERNNDistributionModel`.
        Vendor multidata / cell-type-headed runs still go through PRIEML.
        """
        self._validate_run_dir(model_path)
        self.loaded_model_dir = model_path
        self.selected_cell_type = None

        if self._is_single_head_distribution_run(model_path):
            self.model = self._load_pernn_from_run_dir(model_path)
            self.model_components = None
            self.is_trained = True
            return

        self.model_components = self.prieml_model.build_retrieve_models(model_path)
        self.model = self.model_components
        self.is_trained = True

    @staticmethod
    def _is_single_head_distribution_run(model_path: str) -> bool:
        """True when the run uses generic COMPONENT_NAMES (not decoder_<cell>)."""
        statedict = Path(model_path) / "model_statedict"
        if (statedict / "decoder.pkl").is_file():
            return True
        if any(statedict.glob("decoder_*.pkl")):
            return False
        import os
        import pickle

        options_path = os.path.join(model_path, "config", "exp_options.pkl")
        if not os.path.isfile(options_path):
            return False
        with open(options_path, "rb") as handle:
            options = pickle.load(handle)
        return (
            str(options.get("model_name", "")) == "PE_RNN_distribution"
            and not (options.get("datasets_name") or [])
        )

    def _load_pernn_from_run_dir(self, model_path: str) -> PERNNDistributionModel:
        import os

        mconfig_dir = os.path.join(model_path, "config")
        mconfig, options = self.prieml_model._load_model_config(mconfig_dir)
        model_config = mconfig["model_config"]
        hyperparameters = {
            "embed_dim": int(model_config.embed_dim),
            "num_hidden_layers": int(model_config.num_hidden_layers),
            "bidirection": bool(model_config.bidirection),
            "p_dropout": float(model_config.p_dropout),
        }
        model = build_pernn_distribution_model(
            hyperparameters,
            seqlevel_featdim=int(options.get("seqlevel_featdim", 0)),
            num_outcomes=int(options.get("num_outcomes", 1)),
            device=self.device,
        )
        statedict_dir = os.path.join(model_path, "model_statedict")
        model.load_vendor_statedict(statedict_dir, device=self.device)
        model.eval()
        return model

    def _predict_pernn_dataframe(
        self,
        df: pd.DataFrame,
        y_ref: List[str],
        *,
        batch_size: int = 256,
    ) -> pd.DataFrame:
        """Score a dataframe with the in-memory :class:`PERNNDistributionModel`."""
        if not isinstance(self.model, PERNNDistributionModel):
            raise ValueError("PERNNDistributionModel is not loaded.")
        native = self._to_pridict_dataframe(df)
        dtensor, _ = self._build_datatensor(native, y_ref)
        partition = {"eval": dtensor}
        loaders, _, _, _ = construct_load_dataloaders(
            partition,
            ["eval"],
            {"batch_size": int(batch_size), "num_workers": 0},
            wrk_dir=None,
        )
        loader = loaders["eval"]
        pred_chunks: List[np.ndarray] = []
        true_chunks: List[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                logits, target = self.model.forward_batch(
                    batch,
                    device=self.device,
                    requires_grad=False,
                )
                # Distribution decoder emits log-probs under KLDloss/CEloss.
                probs = torch.exp(logits)
                pred_chunks.append(probs.detach().cpu().numpy())
                true_chunks.append(target.detach().cpu().numpy())
        if not pred_chunks:
            return pd.DataFrame()
        pred = np.concatenate(pred_chunks, axis=0)
        true = np.concatenate(true_chunks, axis=0)
        out: Dict[str, Any] = {}
        for idx, outcome in enumerate(y_ref):
            out[f"true_{outcome}"] = true[:, idx]
            out[f"pred_{outcome}"] = pred[:, idx]
        return pd.DataFrame(out)
    def prepare_data(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Prepare data in PRIDICT2 format
        
        Args:
            df: DataFrame with standard pegRNA features
            **kwargs: Additional arguments
                - y_ref: List of target columns (default: ['averageedited', 'averageunedited', 'averageindel'])
                - batch_size: Batch size for DataLoader (default: 500)
                - cell_types: List of cell types for each sample
                
        Returns:
            DataLoader ready for PRIDICT2 prediction
        """
        df = self._to_pridict_dataframe(df)
        y_ref = kwargs.get('y_ref', self._default_outcomes())
        batch_size = kwargs.get('batch_size', 500)
        cell_types = kwargs.get('cell_types')
        if cell_types is None and self.selected_cell_type:
            cell_types = [self.selected_cell_type]
        elif cell_types is None and self.is_trained and self.loaded_model_dir:
            cell_types = self._cell_types_from_loaded_config()
        model_name = self.model_name_str or kwargs.get('model_name', 'base_390k')

        # Prepare data using PRIDICT2's preprocessing pipeline
        dloader = self.prieml_model.prepare_data(
            df=df,
            model_name=model_name,
            cell_types=cell_types or [],
            y_ref=y_ref,
            batch_size=batch_size
        )
        
        return dloader
    
    def predict(self, data: Any, batch_size: int = 500) -> List[float]:
        """
        Make predictions using PRIDICT model.
        
        Args:
            data: Native PRIDICT DataFrame (ensemble-trained single-head) or a
                DataLoader from :meth:`prepare_data` (vendor multi-head).
            batch_size: Batch size for single-head PERNN inference.
            
        Returns:
            List of intended edit predictions.
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        if isinstance(self.model, PERNNDistributionModel):
            if not isinstance(data, pd.DataFrame):
                raise ValueError(
                    "PERNNDistributionModel predictions require a DataFrame input "
                    "(not a vendor ConcatDataLoader)."
                )
            pred_df = self._predict_pernn_dataframe(
                data, ["averageedited"], batch_size=batch_size
            )
            return pred_df["pred_averageedited"].astype(float).tolist()
        pred_df = self._predict_from_loaded_or_current_model(
            dloader=data,
            y_ref=["averageedited"],
        )
        return pred_df["pred_averageedited"].astype(float).tolist()

    def predict_distribution(
        self,
        data: Any,
        outcomes: Optional[List[str]] = None,
    ) -> List[List[float]]:
        """Predict one or multiple PRIDICT outcomes in batch."""
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        y_ref = outcomes or self._default_outcomes()
        pred_df = self._predict_from_loaded_or_current_model(dloader=data, y_ref=y_ref)
        return pred_df[[f"pred_{outcome}" for outcome in y_ref]].values.tolist()
    
    def predict_single_outcome(self, data: Any, outcome: str = 'averageedited') -> List[float]:
        """
        Make predictions for a single outcome
        
        Args:
            data: DataLoader from prepare_data()
            outcome: Which outcome to predict ('averageedited', 'averageunedited', or 'averageindel')
            
        Returns:
            List of predicted values for the specified outcome
        """
        if not self.is_trained:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        valid_outcomes = ['averageedited', 'averageunedited', 'averageindel']
        if outcome not in valid_outcomes:
            raise ValueError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")
        
        pred_df = self._predict_from_loaded_or_current_model(dloader=data, y_ref=[outcome])
        
        return pred_df[f'pred_{outcome}'].tolist()

    def _build_datatensor(self, df: pd.DataFrame, y_ref: List[str]) -> tuple[Any, List[str]]:
        df = self._prepare_pridict_frame(df)
        norm_cols, proc, init, n_init, mut, n_mut = self.prieml_model._process_df(df)
        dtensor = self.prieml_model._construct_datatensor(
            norm_cols, proc, init, n_init, mut, n_mut, y_ref=y_ref
        )
        return dtensor, list(norm_cols or [])

    @staticmethod
    def _prepare_pridict_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure unique ``seq_id`` and string edit-position lists for vendor preprocess.

        Merged PE-DB exports can collide on per-sheet ``seq_*`` IDs; PRIDICT2
        ``groupby(seq_id)`` then yields multi-row groups and crashes on
        ``deepeditposition_lst.strip``.
        """
        out = df.copy() if "seq_id" not in df.columns else df
        if "seq_id" not in out.columns:
            out["seq_id"] = [f"seq_{i}" for i in range(len(out))]
        elif not out["seq_id"].astype(str).is_unique:
            out = out.copy()
            out["seq_id"] = [f"seq_{i}" for i in range(len(out))]
        if "deepeditposition_lst" in out.columns:
            col = out["deepeditposition_lst"]
            if any(not isinstance(v, str) for v in col.to_numpy()):
                if out is df:
                    out = out.copy()
                out["deepeditposition_lst"] = [
                    v if isinstance(v, str) else str(v) for v in out["deepeditposition_lst"]
                ]
        return out

    @staticmethod
    def _seqlevel_featdim_from_datatensor(dtensor: Any) -> int:
        """Return MLP input width from the built tensor (not continuous-only norm cols).

        Vendor tensors append correction-type and Tm-NaN indicators to the
        continuous columns, so ``len(norm_cols)`` under-counts the real feature dim.
        """
        colnames = getattr(dtensor, "seqlevel_feat_colnames", None)
        if colnames is not None:
            return int(len(colnames))
        feat = getattr(dtensor, "seqlevel_feat", None)
        if feat is not None and hasattr(feat, "shape") and len(feat.shape) >= 2:
            return int(feat.shape[-1])
        raise ValueError(
            "PRIDICT2 datatensor is missing seqlevel_feat_colnames / seqlevel_feat"
        )

    @staticmethod
    def _resolve_pretrained_statedict_dir(weights_name: str) -> str:
        """Return pretrained statedict root for fine-tuning from a weight selection.

        Ensemble-registered runs store ``model_statedict`` directly under the run
        directory. Vendor PRIEML trees use ``<exp>/train_val/run_N/model_statedict``;
        for those, return the experiment root so :func:`vendor_state_dict_path` can
        append the train_val path.
        """
        run_dir, _ = PRIDICT2ModelWrapper.resolve_weight_selection(weights_name)
        direct = run_dir / "model_statedict"
        if direct.is_dir():
            return str(direct)
        return str(run_dir.parent.parent)

    @staticmethod
    def _architecture_from_pretrained_weights(weights_name: str) -> Dict[str, Any]:
        """Read backbone architecture from a registered/vendor PRIDICT2 run."""
        run_dir, _ = PRIDICT2ModelWrapper.resolve_weight_selection(weights_name)
        config_dir = run_dir / "config"
        if not config_dir.is_dir():
            return {}
        from pridict2.pridict.pridictv2.predict_outcomedistrib import PRIEML_Model

        # Lightweight config load without constructing a full wrapper.
        reader = PRIEML_Model(device=torch.device("cpu"), wsize=20)
        mconfig, _options = reader._load_model_config(str(config_dir))
        model_config = mconfig["model_config"]
        return {
            "embed_dim": int(model_config.embed_dim),
            "num_hidden_layers": int(model_config.num_hidden_layers),
            "bidirection": bool(model_config.bidirection),
            "p_dropout": float(model_config.p_dropout),
        }

    def _resolve_train_statedict_dir(
        self, hyperparameters: Dict[str, Any]
    ) -> Optional[str]:
        if not bool(hyperparameters.get("load_pretrained", False)):
            return None
        weights = hyperparameters.get("weights")
        if not weights:
            return None
        return self._resolve_pretrained_statedict_dir(str(weights))

    def _build_trf_tup(
        self,
        hyperparameters: Dict[str, Any],
        *,
        batch_size: int,
        num_epochs: int,
    ) -> tuple[Any, ...]:
        from torch import nn

        if "trf_tup" in hyperparameters:
            return hyperparameters["trf_tup"]
        embed_dim = int(hyperparameters.get("embed_dim", 64))
        # Vendor stack assembly derives z_dim from embed widths.
        z_dim = _compute_z_dim(embed_dim)
        num_hidden_layers = int(hyperparameters.get("num_hidden_layers", 1))
        bidirection = bool(hyperparameters.get("bidirection", True))
        p_dropout = float(
            hyperparameters.get(
                "p_dropout",
                hyperparameters.get("dropout", 0.1),
            )
        )
        l2_reg = float(hyperparameters.get("weight_decay", 1e-4))
        return (
            embed_dim,
            z_dim,
            num_hidden_layers,
            bidirection,
            p_dropout,
            nn.GRU,
            nn.ReLU(),
            l2_reg,
            batch_size,
            num_epochs,
        )

    def _build_experiment_options(
        self,
        hyperparameters: Dict[str, Any],
        *,
        seqlevel_featdim: int,
        y_ref: List[str],
    ) -> Dict[str, Any]:
        experiment_options = {
            "experiment_desc": str(
                hyperparameters.get("experiment_desc", "pe_ensemble_pridict_train")
            ),
            "model_name": "PE_RNN_distribution",
            "annot_embed": _ANNOT_EMBED,
            "assemb_opt": _ASSEMB_OPT,
            # Single-head layout (matches PERNNDistributionModel COMPONENT_NAMES).
            # Vendor PRIEML multidata load requires datasets_name; we load our
            # own artifacts via PERNNDistributionModel instead.
            "datasets_name": [],
            "separate_attention_layers": False,
            "separate_seqlevel_embedder": False,
            "seqlevel_featdim": int(seqlevel_featdim),
            "num_outcomes": int(len(y_ref)),
        }
        if bool(hyperparameters.get("freezing", False)):
            experiment_options["freezing"] = True
            experiment_options["trainable_layernames"] = list(
                hyperparameters.get("trainable_layernames", ["decoder"])
            )
        return experiment_options

    def _metrics_from_validation_predictions(
        self,
        val_df: pd.DataFrame,
        y_ref: List[str],
    ) -> Dict[str, float]:
        if isinstance(self.model, PERNNDistributionModel):
            pred_df = self._predict_pernn_dataframe(val_df, y_ref)
        else:
            prepared_val = self.prepare_data(val_df, y_ref=y_ref)
            pred_df = self._predict_from_loaded_or_current_model(
                dloader=prepared_val,
                y_ref=y_ref,
            )
        fold_metrics: Dict[str, float] = {}
        for outcome in y_ref:
            true_col = f"true_{outcome}"
            pred_col = f"pred_{outcome}"
            if true_col not in pred_df.columns or pred_col not in pred_df.columns:
                continue
            y_true = pd.to_numeric(pred_df[true_col], errors="coerce").to_numpy(dtype=np.float64)
            y_pred = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
            fold_metrics.update(regression_metrics(y_true, y_pred, prefix=outcome))
        return fold_metrics

    def _run_train_val_once(
        self,
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        y_ref: List[str],
        hyperparameters: Dict[str, Any],
        output_dir: str,
        run_suffix: str,
        progress_log: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        from pridict2.pridict.pridictv2.run_workflow import build_config_map

        # When fine-tuning, inherit backbone architecture from the pretrained run.
        build_hparams = dict(hyperparameters)
        weights_name = hyperparameters.get("weights")
        if bool(hyperparameters.get("load_pretrained", False)) and weights_name:
            inherited = self._architecture_from_pretrained_weights(str(weights_name))
            build_hparams.update(inherited)

        batch_size = int(build_hparams.get("batch_size", 128))
        num_epochs = int(build_hparams.get("num_epochs", 20))
        trf_tup = self._build_trf_tup(
            build_hparams,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        dtensor_train, _ = self._build_datatensor(train_df, y_ref)
        dtensor_val, _ = self._build_datatensor(val_df, y_ref)
        seqlevel_featdim = self._seqlevel_featdim_from_datatensor(dtensor_train)
        experiment_options = self._build_experiment_options(
            build_hparams,
            seqlevel_featdim=seqlevel_featdim,
            y_ref=y_ref,
        )
        config_map = build_config_map(
            trf_tup,
            experiment_options,
            loss_func=str(build_hparams.get("loss_func", "MSEloss")),
        )
        train_loader, val_loader = build_pridict_dataloaders(
            train_dataset=dtensor_train,
            val_dataset=dtensor_val,
            batch_size=batch_size,
        )

        model = build_pernn_distribution_model(
            build_hparams,
            seqlevel_featdim=seqlevel_featdim,
            num_outcomes=len(y_ref),
            device=self.device,
        )
        statedict_dir = self._resolve_train_statedict_dir(hyperparameters)
        pretrained_path = vendor_state_dict_path(statedict_dir)
        if pretrained_path:
            model.load_vendor_statedict(pretrained_path, device=self.device)

        train_metrics = train_pridict2_with_lightning(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            hyperparameters=build_hparams,
            device=self.device,
            loss_func=str(build_hparams.get("loss_func", "MSEloss")),
            progress_log=progress_log,
            cancel_check=cancel_check,
        )

        run_output_dir = f"{output_dir}/{run_suffix}"
        model_dir = Path(f"{run_output_dir}/train_val/run_0")
        save_pridict_run_artifacts(
            model_dir=model_dir,
            model=model,
            config_map=config_map,
            best_epoch=int(train_metrics["best_epoch"]),
        )

        # Keep the trained single-head model in memory; vendor PRIEML load expects
        # multidata datasets_name / decoder_<cell> files and would crash here.
        self.model = model
        self.model_components = None
        self.loaded_model_dir = str(model_dir)
        self.is_trained = True
        fold_metrics = self._metrics_from_validation_predictions(val_df, y_ref)
        return {
            "output_dir": run_output_dir,
            "model_dir": str(model_dir),
            "num_train_rows": len(train_df),
            "num_val_rows": len(val_df),
            "metrics": fold_metrics,
            "training_metrics": train_metrics,
        }

    def train(self, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None,
              hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train PRIDICT2 model
        
        Args:
            train_data: Training DataFrame
            val_data: Validation DataFrame
            hyperparameters: Training hyperparameters
            
        Returns:
            Dictionary with training results
        """
        from ..training.progress_log import take_job_training_callbacks

        hyperparameters, progress_log, cancel_check = take_job_training_callbacks(
            hyperparameters
        )
        train_native, val_native = resolve_train_val_from_splits(train_data, val_data)
        train_df = self._to_pridict_dataframe(train_native)
        val_df = self._to_pridict_dataframe(val_native)
        y_ref = list(
            hyperparameters.get("y_ref", self._default_outcomes()) or self._default_outcomes()
        )

        output_dir = str(hyperparameters.get("output_dir", "artifacts/pridict2_train"))
        fold_reports: List[Dict[str, Any]] = []

        if val_data is None and has_assigned_cv_folds(train_data):
            for fold_idx, (fold_label, fold_train_native, fold_val_native) in enumerate(
                iter_assigned_cv_folds(train_data)
            ):
                if cancel_check is not None:
                    cancel_check()
                fold_train_df = self._to_pridict_dataframe(fold_train_native)
                fold_val_df = self._to_pridict_dataframe(fold_val_native)
                report = self._run_train_val_once(
                    train_df=fold_train_df,
                    val_df=fold_val_df,
                    y_ref=y_ref,
                    hyperparameters=hyperparameters,
                    output_dir=output_dir,
                    run_suffix=f"cv_{fold_label}",
                    progress_log=progress_log,
                    cancel_check=cancel_check,
                )
                fold_reports.append({"fold": fold_idx, "fold_label": fold_label, **report})

        if cancel_check is not None:
            cancel_check()
        run_report = self._run_train_val_once(
            train_df=train_df,
            val_df=val_df,
            y_ref=y_ref,
            hyperparameters=hyperparameters,
            output_dir=output_dir,
            run_suffix="final",
            progress_log=progress_log,
            cancel_check=cancel_check,
        )
        result: Dict[str, Any] = {
            "status": "success",
            "output_dir": run_report["output_dir"],
            "model_dir": run_report["model_dir"],
            "num_train_rows": int(run_report["num_train_rows"]),
            "num_val_rows": int(run_report["num_val_rows"]),
            "outcomes": y_ref,
            "validation_metrics": run_report["metrics"],
        }
        if fold_reports:
            result["cross_validation"] = fold_reports
        return result
    
    def evaluate(self, test_data: pd.DataFrame, weights: str) -> Dict[str, float]:
        """
        Evaluate PRIDICT2 model on all three outcomes using a registered weight set.

        Args:
            test_data: Test DataFrame with true labels
            weights: Registered weight set ID (see :meth:`list_available_weights`).

        Returns:
            Dictionary with evaluation metrics for each outcome
        """
        if not weights or not str(weights).strip():
            raise ValueError(
                "weights is required for evaluate(). "
                f"Available: {self.list_available_weights()}"
            )
        self.load_weights_by_name(weights)
        
        test_df = self._to_pridict_dataframe(test_data)
        outcomes = [o for o in self._default_outcomes() if o in test_df.columns]
        if not outcomes:
            outcomes = ["averageedited"]

        if isinstance(self.model, PERNNDistributionModel):
            pred_df = self._predict_pernn_dataframe(test_df, outcomes)
        else:
            dloader = self.prepare_data(test_df, y_ref=outcomes)
            pred_df = self._predict_from_loaded_or_current_model(
                dloader=dloader, y_ref=outcomes
            )
            if self.selected_cell_type and "dataset_name" in pred_df.columns:
                pred_df = pred_df[
                    pred_df["dataset_name"] == self.selected_cell_type
                ].reset_index(drop=True)
        
        primary_outcome = "averageedited" if "averageedited" in outcomes else outcomes[0]
        results: Dict[str, float] = {}
        for outcome in outcomes:
            y_true = pred_df[f"true_{outcome}"].values
            y_pred = pred_df[f"pred_{outcome}"].values
            results.update(regression_metrics(y_true, y_pred, prefix=outcome))
            if outcome == primary_outcome:
                results.update(regression_metrics(y_true, y_pred))
        results["n_samples"] = int(len(pred_df))
        return results
    
    def save_model(self, model_path: str) -> None:
        """Copy the loaded PRIDICT2 run directory into ``model_path``."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        if not self.loaded_model_dir:
            raise ValueError("No loaded PRIDICT2 run directory to save.")

        import os
        import shutil

        src = Path(self.loaded_model_dir)
        os.makedirs(model_path, exist_ok=True)
        for name in ("model_statedict", "config"):
            src_sub = src / name
            if src_sub.is_dir():
                dest_sub = Path(model_path) / name
                if dest_sub.exists():
                    shutil.rmtree(dest_sub)
                shutil.copytree(src_sub, dest_sub)
    
    def save_to_registry(self, dest_dir) -> str:
        self.save_model(str(dest_dir))
        return "pridict2_run"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        info = super().get_model_info()
        info.update({
            'model_name': self.model_name_str,
            'wsize': self.wsize,
            'available_weights': self.list_available_weights(),
            'outcomes': self._default_outcomes(),
            'supports_standardized_input': True,
            'description': 'PRIDICT2 model for predicting outcome distribution (edited/unedited/indel)'
        })
        if self.is_trained and self.loaded_model_dir:
            info['cell_types_from_config'] = self._cell_types_from_loaded_config()
        if self.selected_cell_type:
            info['selected_cell_type'] = self.selected_cell_type
        return info
