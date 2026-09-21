"""Behavioral checks for OPED optimization and portable trained models."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from pe_ensemble.models.oped_wrapper import (
    OPEDModelWrapper, _OPEDEncodedDataset, _OPEDLightningRegressor,
)


@pytest.fixture
def encoded():
    rng = np.random.default_rng(3)
    data = {}
    for order, suffix in [(1, ''), (2, '_o2'), (3, '_o3')]:
        for name, length in [('Target', 12), ('PBS', 6), ('RT', 9)]:
            data[name + suffix] = list(rng.integers(1, 4**order + 1, (7, length-order+1)))
    data['Efficiency'] = np.linspace(0.1, 0.7, 7)
    return pd.DataFrame(data)


def small_hparams():
    return dict(embedding_size=8, nhead=2, hidden_size=[16]*3,
                num_encoder_layers=[1]*3, drop_out=0.0, batch_size=4,
                epoch_num=2, num_workers=0, lr=0.001, scheduler='none',
                early_stopping_patience=0)


def test_saved_custom_heads_preserve_predictions(tmp_path, encoded):
    wrapper = OPEDModelWrapper(device=torch.device('cpu'))
    wrapper.model = wrapper._build_model_from_hparams(small_hparams())
    wrapper.is_trained = True
    before = wrapper._predict_encoded_df(wrapper.model, encoded)
    wrapper.save_model(str(tmp_path / 'weights.pt'))
    restored = OPEDModelWrapper(device=torch.device('cpu'))
    restored.load_model(str(tmp_path / 'weights.pt'))
    assert restored.model.nhead == 2
    np.testing.assert_allclose(restored._predict_encoded_df(restored.model, encoded), before, atol=1e-6)


def test_training_aliases_and_freezing_are_honored(monkeypatch, encoded):
    wrapper = OPEDModelWrapper(device=torch.device('cpu'))
    captured = []
    def loop(model, train_df, val_df, hparams, **kwargs):
        captured.append((copy.deepcopy(hparams), model))
        return model, dict(history=[], best_epoch=0, best_val_loss=1., n_epochs_ran=1)
    monkeypatch.setattr(wrapper, '_run_training_loop', loop)
    hp = dict(embedding_size=8, nhead=2, ffn_dim=16, encoder_layers=1,
              dropout=0., epochs=2, num_workers=0)
    wrapper.train(encoded.iloc[:4], encoded.iloc[4:], hp, freezing=True)
    params, model = captured[0]
    assert params['epoch_num'] == 2
    assert len(model.encoder_decoder[0].layers) == 1
    assert model.encoder_decoder[0].layers[0].linear1.out_features == 16
    assert model.dropout == 0.
    assert not model.embedding[0].weight.requires_grad
    assert all(p.requires_grad for p in model.fully_connected_layers.parameters())


def test_real_training_updates_weights_and_reports_restored_validation(encoded):
    torch.set_num_threads(1)
    wrapper = OPEDModelWrapper(device=torch.device('cpu'))
    hp = small_hparams()
    model = wrapper._build_model_from_hparams(hp)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    model, report = wrapper._run_training_loop(model, encoded.iloc[:4], encoded.iloc[4:], hp)
    assert any(not torch.equal(value, initial[key]) for key, value in model.state_dict().items())
    preds = wrapper._predict_encoded_df(model, encoded.iloc[4:])
    expected = np.mean((preds-encoded.Efficiency.iloc[4:].to_numpy())**2)
    assert report['best_val_loss'] == pytest.approx(expected, rel=1e-5)
    assert report['n_epochs_ran'] == 2


def test_lightning_step_matches_explicit_adam_update(encoded):
    torch.set_num_threads(1)
    hp = small_hparams()
    model = OPEDModelWrapper._build_model_from_hparams(hp)
    reference = copy.deepcopy(model)
    module = _OPEDLightningRegressor(model, hp)
    module.log = lambda *args, **kwargs: None
    loader = torch.utils.data.DataLoader(_OPEDEncodedDataset(encoded.iloc[:4]), batch_size=4)
    batch = next(iter(loader))
    optimizer = module.configure_optimizers()
    optimizer.zero_grad()
    module.training_step(batch, 0).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer_ref = torch.optim.Adam(reference.parameters(), lr=hp['lr'])
    optimizer_ref.zero_grad()
    predictions, _ = reference(batch[0])
    torch.nn.functional.mse_loss(predictions.squeeze(-1), batch[1]).backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
    optimizer_ref.step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['merge', 'replace'])
def test_api_presets_do_not_shadow_user_aliases(tmp_path, mode):
    from pe_ensemble.training.hyperparameter_presets import resolve_hyperparameters
    resolved = resolve_hyperparameters('oped', preset_root=tmp_path, mode=mode,
                                      user_overrides=dict(epochs=2, ffn_dim=16,
                                                          encoder_layers=1, dropout=0.))
    hp = resolved.hyperparameters
    assert hp['epoch_num'] == 2
    assert hp['hidden_size'] == [16]*3
    assert hp['num_encoder_layers'] == [1]*3
    assert hp['drop_out'] == 0.


def test_fine_tune_reports_loaded_architecture_and_freezes_backbone(tmp_path, encoded):
    torch.set_num_threads(1)
    from oped.pegRNA_PredictingCodes.train_model import TransformerEncoderDecoderModelOrder3
    source = OPEDModelWrapper(device=torch.device('cpu'))
    source.model = TransformerEncoderDecoderModelOrder3(
        embedding_size=8, nhead=2, hidden_size=[16]*3,
        num_encoder_layers=[1]*3, dropout=0., hidden_size_fully=[8])
    source.is_trained = True
    source.save_model(str(tmp_path/'weights.pt'))
    backbone = {n: p.detach().clone() for n,p in source.model.named_parameters()
                if not n.startswith('fully_connected_layers.')}
    original_head = {n:p.detach().clone() for n,p in source.model.fully_connected_layers.named_parameters()}
    model = OPEDModelWrapper(device=torch.device('cpu'))
    report = model.train(encoded.iloc[:4], encoded.iloc[4:],
                         dict(load_pretrained=True, weights=str(tmp_path), freezing=True,
                              epoch_num=2, batch_size=4, num_workers=0,
                              lr=.001, scheduler='none', early_stopping_patience=0))
    assert report['architecture']['kind'] == 'encoder_decoder'
    assert report['hyperparameters']['nhead'] == 2
    assert report['hyperparameters']['num_encoder_layers'] == [1]*3
    assert report['hyperparameters']['hidden_size_fully'] == [8]
    for name, param in model.model.named_parameters():
        if name in backbone:
            torch.testing.assert_close(param, backbone[name], rtol=0, atol=0)
    assert any(not torch.equal(p, original_head[n])
               for n,p in model.model.fully_connected_layers.named_parameters())


def test_short_kmer_channels_have_finite_training_gradients():
    hp = small_hparams()
    model = OPEDModelWrapper._build_model_from_hparams(hp)
    native = pd.DataFrame({'Target(47bp)': ['ACGT'*12]*2,
                           'PBS': ['A', 'ACGT'], 'RT': ['ACGTACGT']*2})
    frame = OPEDModelWrapper._to_oped_numeric_df(native)
    frame['Efficiency'] = [.2, .4]
    inputs, targets = next(iter(torch.utils.data.DataLoader(_OPEDEncodedDataset(frame), batch_size=2)))
    preds, _ = model(inputs)
    torch.nn.functional.mse_loss(preds.squeeze(-1), targets).backward()
    assert torch.isfinite(preds).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_scratch_default_matches_vendor_decoder_and_hpo_space():
    from oped.pegRNA_PredictingCodes.train_model import TransformerEncoderDecoderModelOrder3
    from pe_ensemble.training.search_spaces import get_search_space
    from pe_ensemble.training.model_baselines import model_baseline_hyperparameters
    model = OPEDModelWrapper._build_model_from_hparams(small_hparams())
    assert isinstance(model, TransformerEncoderDecoderModelOrder3)
    assert isinstance(model.encoder_decoder[1].layers[0], torch.nn.TransformerDecoderLayer)
    assert get_search_space('oped').fixed['model_variant'] == 'encoder_decoder'
    assert model_baseline_hyperparameters('oped')['model_variant'] == 'encoder_decoder'
