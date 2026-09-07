"""HTTP smoke tests for every PE Ensemble FastAPI route.

Job execution is not run: the device scheduler is stubbed so POST /train,
/tune, /evaluate and /ensemble only persist queued job state. Plugin
validation is stubbed the same way. Vendor weight listing still reads the
real registry so the catalog contract is exercised.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.smoke

_DUMMY_ROOT = Path(__file__).resolve().parents[3] / "testdata" / "plugins" / "dummy_model"
_OPED_WEIGHTS = "pegRNA_Model_Merged_saved.order3_decoder_weights"

TRAIN_BODY = {
    "model_name": "deepprime",
    "dataset_source": "pe-db",
    "dataset_name": "library2",
    "device": "cpu",
}
TUNE_BODY = {"training": TRAIN_BODY, "n_trials": 1, "no_write_preset": True}
EVAL_BODY = {
    "model_name": "deepprime",
    "weights": "DeepPrime_base",
    "device": "cpu",
}
ENSEMBLE_BODY = {
    "ensemble_name": "smoke-ensemble",
    "combine": "mean",
    "members": [
        {"model_name": "deepprime", "weights": "DeepPrime_base"},
        {"model_name": "oped", "weights": _OPED_WEIGHTS},
    ],
    "device": "cpu",
}


class _QueueOnlyScheduler:
    def submit_training(self, *args, **kwargs):
        return None

    def submit_tuning(self, *args, **kwargs):
        return None

    def submit_evaluation(self, *args, **kwargs):
        return None

    def submit_ensemble(self, *args, **kwargs):
        return None

    def device_snapshot(self):
        return []


class _DummyModel:
    def load_weights_by_name(self, weight_id):
        return None

    def load_model(self):
        return None


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


def _drop_task(coro):
    close = getattr(coro, "close", None)
    if close is not None:
        close()
    return None


def _http_routes(app) -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if not methods or not path:
            continue
        for method in methods:
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.add((method, path))
    return routes


def _stub_queue_validation(name: str) -> dict[str, Any]:
    from app.plugins.validation_jobs import create_job, get_job

    job_id = create_job(name)
    manifest = get_job(job_id)
    return {
        "job_id": job_id,
        "plugin_name": manifest["plugin_name"],
        "status": manifest["status"],
        "message": "Validation job queued",
    }


def _upload_plugin(client: TestClient, name: str = "smoke_dummy") -> str:
    response = client.post(
        "/models/plugins",
        data={
            "name": name,
            "version": "0.1.0",
            "display_name": "Smoke Dummy",
            "wrapper_class": "DummyModelWrapper",
            "weight_format": "dummy_state_dict",
            "output_columns": "feature,Efficiency",
            "required_std_columns": "edit_len,editing_efficiency",
            "label_column": "Efficiency",
            "replace_existing": "true",
        },
        files={
            "convert_file": ("convert.py", (_DUMMY_ROOT / "convert.py").read_bytes(), "text/x-python"),
            "wrapper_file": ("wrapper.py", (_DUMMY_ROOT / "wrapper.py").read_bytes(), "text/x-python"),
        },
    )
    assert response.status_code == 200, response.text
    return name


def _create_train_job(client: TestClient) -> str:
    response = client.post("/train", json=TRAIN_BODY)
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _create_tune_job(client: TestClient) -> str:
    response = client.post("/tune", json=TUNE_BODY)
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _create_eval_job(client: TestClient) -> str:
    response = client.post("/evaluate", json=EVAL_BODY)
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _create_ensemble_job(client: TestClient) -> str:
    response = client.post("/ensemble", json=ENSEMBLE_BODY)
    assert response.status_code == 200, response.text
    return response.json()["job_id"]


def _create_validation_job(client: TestClient, name: str = "smoke_dummy") -> tuple[str, str]:
    _upload_plugin(client, name)
    response = client.post(f"/models/plugins/{name}/validate")
    assert response.status_code == 202, response.text
    return name, response.json()["job_id"]


RouteCall = Callable[[TestClient], Any]

ROUTE_CASES: dict[tuple[str, str], tuple[RouteCall, int]] = {
    ("GET", "/"): (lambda c: c.get("/"), 200),
    ("GET", "/health"): (lambda c: c.get("/health"), 200),
    ("GET", "/openapi.json"): (lambda c: c.get("/openapi.json"), 200),
    ("GET", "/docs"): (lambda c: c.get("/docs"), 200),
    ("GET", "/docs/oauth2-redirect"): (lambda c: c.get("/docs/oauth2-redirect"), 200),
    ("GET", "/redoc"): (lambda c: c.get("/redoc"), 200),
    ("GET", "/data/filter"): (
        lambda c: c.get("/data/filter", params={"format": "std", "split_strategy": "none"}),
        200,
    ),
    ("GET", "/models"): (lambda c: c.get("/models"), 200),
    ("GET", "/models/{model_name}/training-presets"): (
        lambda c: c.get("/models/deepprime/training-presets"),
        200,
    ),
    ("GET", "/models/{model_name}/weights"): (
        lambda c: c.get("/models/deepprime/weights"),
        200,
    ),
    ("POST", "/predict"): (
        lambda c: c.post(
            "/predict",
            json={
                "model_name": "deepprime",
                "sequences": ["ACGT"],
                "weights": "DeepPrime_base",
                "device": "cpu",
            },
        ),
        200,
    ),
    ("GET", "/devices"): (lambda c: c.get("/devices"), 200),
    ("GET", "/train/devices"): (lambda c: c.get("/train/devices"), 200),
    ("POST", "/train"): (lambda c: c.post("/train", json=TRAIN_BODY), 200),
    ("GET", "/train/jobs"): (lambda c: c.get("/train/jobs"), 200),
    ("GET", "/train/status/{job_id}"): (
        lambda c: c.get(f"/train/status/{_create_train_job(c)}"),
        200,
    ),
    ("DELETE", "/train/jobs/{job_id}"): (
        lambda c: c.delete(f"/train/jobs/{_create_train_job(c)}"),
        202,
    ),
    ("GET", "/train/logs/{job_id}"): (
        lambda c: c.get(f"/train/logs/{_create_train_job(c)}"),
        200,
    ),
    ("POST", "/tune"): (lambda c: c.post("/tune", json=TUNE_BODY), 200),
    ("GET", "/tune/jobs"): (lambda c: c.get("/tune/jobs"), 200),
    ("GET", "/tune/status/{job_id}"): (
        lambda c: c.get(f"/tune/status/{_create_tune_job(c)}"),
        200,
    ),
    ("DELETE", "/tune/jobs/{job_id}"): (
        lambda c: c.delete(f"/tune/jobs/{_create_tune_job(c)}"),
        202,
    ),
    ("GET", "/tune/logs/{job_id}"): (
        lambda c: c.get(f"/tune/logs/{_create_tune_job(c)}"),
        200,
    ),
    ("GET", "/tune/devices"): (lambda c: c.get("/tune/devices"), 200),
    ("POST", "/evaluate"): (lambda c: c.post("/evaluate", json=EVAL_BODY), 200),
    ("GET", "/evaluate/jobs"): (lambda c: c.get("/evaluate/jobs"), 200),
    ("GET", "/evaluate/status/{job_id}"): (
        lambda c: c.get(f"/evaluate/status/{_create_eval_job(c)}"),
        200,
    ),
    ("DELETE", "/evaluate/jobs/{job_id}"): (
        lambda c: c.delete(f"/evaluate/jobs/{_create_eval_job(c)}"),
        202,
    ),
    ("GET", "/evaluate/logs/{job_id}"): (
        lambda c: c.get(f"/evaluate/logs/{_create_eval_job(c)}"),
        200,
    ),
    ("GET", "/evaluate/devices"): (lambda c: c.get("/evaluate/devices"), 200),
    ("GET", "/ensemble/methods"): (lambda c: c.get("/ensemble/methods"), 200),
    ("POST", "/ensemble"): (lambda c: c.post("/ensemble", json=ENSEMBLE_BODY), 200),
    ("GET", "/ensemble/jobs"): (lambda c: c.get("/ensemble/jobs"), 200),
    ("GET", "/ensemble/status/{job_id}"): (
        lambda c: c.get(f"/ensemble/status/{_create_ensemble_job(c)}"),
        200,
    ),
    ("DELETE", "/ensemble/jobs/{job_id}"): (
        lambda c: c.delete(f"/ensemble/jobs/{_create_ensemble_job(c)}"),
        202,
    ),
    ("GET", "/ensemble/logs/{job_id}"): (
        lambda c: c.get(f"/ensemble/logs/{_create_ensemble_job(c)}"),
        200,
    ),
    ("GET", "/ensemble/devices"): (lambda c: c.get("/ensemble/devices"), 200),
    ("GET", "/models/plugins"): (lambda c: c.get("/models/plugins"), 200),
    ("GET", "/models/plugins/{name}"): (
        lambda c: c.get(f"/models/plugins/{_upload_plugin(c)}"),
        200,
    ),
    ("GET", "/models/plugins/{name}/validation.log"): (
        lambda c: c.get(f"/models/plugins/{_upload_plugin(c)}/validation.log"),
        200,
    ),
    ("POST", "/models/plugins"): (
        lambda c: c.post(
            "/models/plugins",
            data={
                "name": "smoke_dummy",
                "version": "0.1.0",
                "display_name": "Smoke Dummy",
                "wrapper_class": "DummyModelWrapper",
                "weight_format": "dummy_state_dict",
                "output_columns": "feature,Efficiency",
                "required_std_columns": "edit_len,editing_efficiency",
                "label_column": "Efficiency",
                "replace_existing": "true",
            },
            files={
                "convert_file": (
                    "convert.py",
                    (_DUMMY_ROOT / "convert.py").read_bytes(),
                    "text/x-python",
                ),
                "wrapper_file": (
                    "wrapper.py",
                    (_DUMMY_ROOT / "wrapper.py").read_bytes(),
                    "text/x-python",
                ),
            },
        ),
        200,
    ),
    ("POST", "/models/plugins/{name}/validate"): (
        lambda c: c.post(f"/models/plugins/{_upload_plugin(c)}/validate"),
        202,
    ),
    ("GET", "/models/plugins/{name}/validate/status/{job_id}"): (
        lambda c: (
            lambda pair: c.get(f"/models/plugins/{pair[0]}/validate/status/{pair[1]}")
        )(_create_validation_job(c)),
        200,
    ),
    ("GET", "/models/plugins/{name}/validate/logs/{job_id}"): (
        lambda c: (
            lambda pair: c.get(f"/models/plugins/{pair[0]}/validate/logs/{pair[1]}")
        )(_create_validation_job(c)),
        200,
    ),
    ("DELETE", "/models/plugins/{name}/validate/jobs/{job_id}"): (
        lambda c: (
            lambda pair: c.delete(f"/models/plugins/{pair[0]}/validate/jobs/{pair[1]}")
        )(_create_validation_job(c)),
        202,
    ),
    ("POST", "/models/plugins/{name}/activate"): (
        lambda c: c.post(f"/models/plugins/{_upload_plugin(c)}/activate"),
        400,
    ),
    ("DELETE", "/models/plugins/{name}"): (
        lambda c: c.delete(f"/models/plugins/{_upload_plugin(c)}"),
        200,
    ),
}


@pytest.fixture
def ensemble_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("TRAINING_JOBS_ROOT", str(tmp_path / "train_jobs"))
    monkeypatch.setenv("TUNING_JOBS_ROOT", str(tmp_path / "tune_jobs"))
    monkeypatch.setenv("EVAL_JOBS_ROOT", str(tmp_path / "eval_jobs"))
    monkeypatch.setenv("ENSEMBLE_JOBS_ROOT", str(tmp_path / "ensemble_jobs"))
    monkeypatch.setenv("PLUGIN_VALIDATION_JOBS_ROOT", str(tmp_path / "validation_jobs"))
    monkeypatch.setenv("PLUGINS_ROOT", str(tmp_path / "plugins"))
    monkeypatch.setenv("TRAINING_PRESETS_ROOT", str(tmp_path / "presets"))
    (tmp_path / "plugins").mkdir()

    from app.main import app

    scheduler = _QueueOnlyScheduler()
    monkeypatch.setattr("app.main.get_scheduler", lambda: scheduler)
    monkeypatch.setattr("app.main.ModelFactory.create_model", lambda *a, **k: _DummyModel())
    monkeypatch.setattr(
        "app.main._request_pe_db_filtered",
        lambda params: {
            "status": "success",
            "groups": [],
            "skipped": [],
            "total_records": 0,
            "target_format": "std",
        },
    )
    monkeypatch.setattr("app.plugins.manager.queue_validation", _stub_queue_validation)
    monkeypatch.setattr("app.main.asyncio.create_task", _drop_task)

    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = _noop_lifespan
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.router.lifespan_context = original_lifespan


def test_api_route_inventory_matches_app():
    from app.main import app

    actual = _http_routes(app)
    declared = set(ROUTE_CASES)
    missing = actual - declared
    extra = declared - actual
    assert not missing, f"Add PE Ensemble API smoke coverage for: {sorted(missing)}"
    assert not extra, f"Remove stale PE Ensemble API smoke cases: {sorted(extra)}"


@pytest.mark.parametrize("method,path", sorted(ROUTE_CASES))
def test_api_route_smoke(ensemble_client: TestClient, method: str, path: str):
    call, expected = ROUTE_CASES[(method, path)]
    response = call(ensemble_client)
    assert response.status_code == expected, response.text


def test_root_and_health_payload(ensemble_client: TestClient):
    root = ensemble_client.get("/").json()
    assert root["service"] == "PE Ensemble"
    assert root["data_filter"] == "/data/filter"
    assert ensemble_client.get("/health").json()["status"] == "healthy"


def test_models_include_builtins(ensemble_client: TestClient):
    payload = ensemble_client.get("/models").json()
    names = {entry["name"] for entry in payload["models"]}
    assert {"deepprime", "oped", "pridict2"} <= names


def test_invalid_model_is_rejected(ensemble_client: TestClient):
    assert ensemble_client.get("/models/not-a-model/weights").status_code == 400
    assert ensemble_client.post("/train", json={**TRAIN_BODY, "model_name": "nope"}).status_code == 400


def test_missing_jobs_are_404(ensemble_client: TestClient):
    assert ensemble_client.get("/train/status/missing-job").status_code == 404
    assert ensemble_client.get("/tune/status/missing-job").status_code == 404
    assert ensemble_client.get("/evaluate/status/missing-job").status_code == 404
    assert ensemble_client.get("/ensemble/status/missing-job").status_code == 404


def test_evaluate_requires_weights(ensemble_client: TestClient):
    response = ensemble_client.post(
        "/evaluate",
        json={"model_name": "deepprime", "weights": "", "device": "cpu"},
    )
    assert response.status_code == 422


def test_evaluate_unknown_weights(ensemble_client: TestClient):
    response = ensemble_client.post(
        "/evaluate",
        json={"model_name": "deepprime", "weights": "not-registered", "device": "cpu"},
    )
    assert response.status_code == 400


def test_ensemble_methods_include_mean(ensemble_client: TestClient):
    payload = ensemble_client.get("/ensemble/methods").json()
    methods = {entry["id"] for entry in payload["methods"]}
    assert "mean" in methods
    assert "weighted_mean" in methods


def test_plugin_missing_is_404(ensemble_client: TestClient):
    assert ensemble_client.get("/models/plugins/does_not_exist").status_code == 404
    assert ensemble_client.delete("/models/plugins/does_not_exist").status_code == 404
