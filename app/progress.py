from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


_jobs: dict[str, dict[str, Any]] = {}
_lock = Lock()


def create_job() -> str:
    job_id = uuid4().hex
    now = _now()
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "stage": "queued",
            "percent": 0,
            "message": "任务已创建，等待开始分析。",
            "detail": {},
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }
    return job_id


def update_job(job_id: str, *, stage: str, percent: int, message: str, detail: dict[str, Any] | None = None) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.update(
            {
                "stage": stage,
                "percent": max(0, min(100, int(percent))),
                "message": message,
                "detail": detail or {},
                "updated_at": _now(),
            }
        )


def complete_job(job_id: str, result: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        partial = bool(getattr(result, "evaluation", {}).get("partial_result")) if result is not None else False
        job.update(
            {
                "status": "partial" if partial else "completed",
                "stage": "completed",
                "percent": 100,
                "message": "已返回部分结果。" if partial else "分析完成。",
                "result": result,
                "updated_at": _now(),
            }
        )


def fail_job(job_id: str, error: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.update(
            {
                "status": "failed",
                "stage": "failed",
                "percent": 100,
                "message": "分析失败。",
                "error": error,
                "updated_at": _now(),
            }
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        data = deepcopy(job)
        data.pop("result", None)
        return data


def get_result(job_id: str) -> Any | None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return None
        return job.get("result")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
