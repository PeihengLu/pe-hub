# Vendor evaluation fixtures

Small CSV fixtures (two pegRNA rows each) used by
`services/pe-ensemble/tests/test_vendor_models_evaluation.py`.

- `standardized_small.csv` — shared labels (`editing_efficiency`) for OPED
- `deepprime_small.csv` — DeepPrime native features + `Efficiency`
- `oped_native_small.csv` — OPED sequence columns (`Target(47bp)`, `PBS`, `RT`)
- `pridict2_small.csv` — PRIDICT2 native features + `averageedited`

Regenerate from the standardized source (requires `services/pe-db` on `PYTHONPATH`):

```bash
cd services/pe-db
PYTHONPATH=.:../../packages/pe-common python3 ../../scripts/regenerate_vendor_eval_fixtures.py
```
