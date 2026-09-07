"""HTTP smoke tests for every PE-DB FastAPI route.

Each public (method, path) pair is exercised once so a refactor that drops or
renames an endpoint fails here before it reaches the CLI or PE Hub UI.
Heavy pipeline work (export / convert / parquet load) is stubbed; catalog
reads hit an isolated seeded sqlite database.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Callable

import pandas as pd
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.smoke

RouteCall = Callable[[TestClient], Any]


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


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


ROUTE_CASES: dict[tuple[str, str], tuple[RouteCall, int]] = {
    ("GET", "/"): (lambda c: c.get("/"), 200),
    ("GET", "/health"): (lambda c: c.get("/health"), 200),
    ("GET", "/openapi.json"): (lambda c: c.get("/openapi.json"), 200),
    ("GET", "/docs"): (lambda c: c.get("/docs"), 200),
    ("GET", "/docs/oauth2-redirect"): (lambda c: c.get("/docs/oauth2-redirect"), 200),
    ("GET", "/redoc"): (lambda c: c.get("/redoc"), 200),
    ("GET", "/api/studies"): (lambda c: c.get("/api/studies"), 200),
    ("GET", "/api/datasets"): (
        lambda c: c.get("/api/datasets", params={"study": "deepprime"}),
        200,
    ),
    ("GET", "/api/datasheets"): (lambda c: c.get("/api/datasheets"), 200),
    ("GET", "/api/scaffolds"): (lambda c: c.get("/api/scaffolds"), 200),
    ("GET", "/api/scaffolds/{scaffold_id}"): (lambda c: c.get("/api/scaffolds/1"), 200),
    ("GET", "/api/filter"): (
        lambda c: c.get("/api/filter", params={"study": "deepprime"}),
        200,
    ),
    ("GET", "/api/data"): (
        lambda c: c.get(
            "/api/data",
            params={
                "study": "deepprime",
                "dataset": "library2",
                "cell_line": "HEK293T",
                "pe_system": "PE2max",
                "limit": 1,
            },
        ),
        200,
    ),
    ("GET", "/api/statistics"): (lambda c: c.get("/api/statistics"), 200),
    ("POST", "/api/export"): (
        lambda c: c.post("/api/export", params={"study": "deepprime"}),
        200,
    ),
    ("POST", "/api/convert"): (
        lambda c: c.post(
            "/api/convert",
            params={
                "study": "deepprime",
                "dataset": "library2",
                "cell_line": "HEK293T",
                "pe_system": "PE2max",
            },
        ),
        200,
    ),
    ("POST", "/api/plugins/reload"): (lambda c: c.post("/api/plugins/reload"), 200),
}


@pytest.fixture
def pe_db_client(seeded_catalog, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.main import app

    monkeypatch.setattr(
        "app.main.library_run_export",
        lambda **kwargs: {
            "status": "success",
            "study": kwargs.get("study") or "all",
            "force_reexport": bool(kwargs.get("force_reexport")),
            "standardized": bool(kwargs.get("standardize", True)),
            "datasheets_in_catalog": 0,
        },
    )
    monkeypatch.setattr(
        "app.main.run_convert_sheet",
        lambda **kwargs: {
            "status": "success",
            "message": "stub",
            "records_converted": 0,
            "output_columns": [],
        },
    )
    monkeypatch.setattr(
        "app.main.loader.load_data",
        lambda **kwargs: pd.DataFrame(
            {"edit_len": [1], "editing_efficiency": [0.25]}
        ),
    )

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
    assert not missing, f"Add PE-DB API smoke coverage for: {sorted(missing)}"
    assert not extra, f"Remove stale PE-DB API smoke cases: {sorted(extra)}"


@pytest.mark.parametrize("method,path", sorted(ROUTE_CASES))
def test_api_route_smoke(pe_db_client: TestClient, method: str, path: str):
    call, expected = ROUTE_CASES[(method, path)]
    response = call(pe_db_client)
    assert response.status_code == expected, response.text


def test_root_lists_catalog_endpoints(pe_db_client: TestClient):
    payload = pe_db_client.get("/").json()
    assert payload["name"] == "PE Database API"
    assert payload["endpoints"]["filter"] == "/api/filter"


def test_health_reports_isolated_catalog(pe_db_client: TestClient, seeded_catalog):
    payload = pe_db_client.get("/health").json()
    assert payload["status"] == "healthy"
    assert payload["catalog_database_exists"] is True
    assert str(seeded_catalog) in payload["catalog_database"]


def test_studies_include_builtins(pe_db_client: TestClient):
    names = {row["name"] for row in pe_db_client.get("/api/studies").json()}
    assert "deepprime" in names
    assert "pridict1" in names


def test_scaffold_not_found(pe_db_client: TestClient):
    response = pe_db_client.get("/api/scaffolds/99999")
    assert response.status_code == 404


def test_filter_catalog_only_returns_datasheets(pe_db_client: TestClient):
    payload = pe_db_client.get("/api/filter").json()
    assert payload["status"] == "success"
    assert payload["format"] is None
    assert payload["datasheets"] == []


def test_filter_format_requires_split_strategy(pe_db_client: TestClient):
    response = pe_db_client.get("/api/filter", params={"format": "std"})
    assert response.status_code == 422
    assert "split_strategy" in response.json()["detail"]


def test_filter_std_none_split_on_empty_catalog(pe_db_client: TestClient):
    response = pe_db_client.get(
        "/api/filter",
        params={"format": "std", "split_strategy": "none", "dataset": "missing"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload.get("total_records", 0) == 0


def test_filter_summary_only_defaults_format(pe_db_client: TestClient):
    response = pe_db_client.get(
        "/api/filter",
        params={"summary_only": True, "split_strategy": "none"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"


def test_data_requires_study(pe_db_client: TestClient):
    response = pe_db_client.get(
        "/api/data",
        params={"dataset": "library2", "cell_line": "HEK293T", "pe_system": "PE2max"},
    )
    assert response.status_code == 422


def test_data_legacy_source_model(pe_db_client: TestClient):
    response = pe_db_client.get(
        "/api/data",
        params={
            "source_model": "dp",
            "dataset": "library2",
            "cell_line": "HEK293T",
            "pe_system": "PE2max",
            "limit": 1,
        },
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["study"] == "deepprime"


def test_data_file_not_found(pe_db_client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "app.main.loader.load_data",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing parquet")),
    )
    response = pe_db_client.get(
        "/api/data",
        params={
            "study": "deepprime",
            "dataset": "library2",
            "cell_line": "HEK293T",
            "pe_system": "PE2max",
        },
    )
    assert response.status_code == 404


def test_statistics_empty_catalog_shape(pe_db_client: TestClient):
    payload = pe_db_client.get("/api/statistics").json()
    assert payload["total_entries"] == 0
    assert payload["total_studies"] == 0
    assert payload["edit_type"] == []


def test_plugins_reload_empty_root(pe_db_client: TestClient):
    payload = pe_db_client.post("/api/plugins/reload").json()
    assert payload["count"] == 0
    assert payload["loaded"] == []
