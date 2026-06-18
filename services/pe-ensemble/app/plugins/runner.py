"""Execute plugin validation jobs on a background worker thread."""
from __future__ import annotations

import logging

from ..compute.job_cancel import JobCancelledError, is_cancel_requested
from .manager import execute_validation
from .validation_jobs import (
    append_log,
    get_job,
    job_log_context,
    mark_cancelled,
    mark_failed,
    mark_running,
    mark_succeeded,
)

logger = logging.getLogger(__name__)


def run_plugin_validation_job(job_id: str) -> None:
    manifest = get_job(job_id)
    plugin_name = manifest["plugin_name"]

    if is_cancel_requested("validate", job_id):
        mark_cancelled(job_id)
        return

    try:
        with job_log_context(job_id):
            mark_running(job_id)
            if is_cancel_requested("validate", job_id):
                raise JobCancelledError(f"Validation job {job_id} cancelled")

            def log_line(message: str) -> None:
                append_log(job_id, message)

            result = execute_validation(plugin_name, log_line=log_line)
            mark_succeeded(job_id, result)
    except JobCancelledError:
        logger.info("Plugin validation job %s cancelled", job_id)
        try:
            mark_cancelled(job_id)
        except FileNotFoundError:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("Plugin validation job %s failed", job_id)
        try:
            current = get_job(job_id)
            if current.get("status") == "running":
                mark_failed(job_id, str(exc))
        except FileNotFoundError:
            pass
