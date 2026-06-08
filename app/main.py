import asyncio
from fastapi import Depends, FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from typing import Any

from .config import Settings, get_settings
from .company_search import search_companies
from .db import SessionLocal, get_db
from .progress import complete_job, create_job, fail_job, get_job, get_result, update_job
from .rag import run_analysis
from .schemas import AnalyzeRequest, AnalyzeResponse, CompanySearchResult
from .secrets import delete_deepseek_api_key, has_deepseek_api_key


app = FastAPI(title="News Market Association RAG", version="0.1.0")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/policy")
def policy() -> dict[str, bool | str]:
    return {
        "purpose": "association analysis only",
        "not_investment_advice": True,
        "no_price_prediction": True,
        "no_trading_signal": True,
        "no_causal_claim": True,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
    db: Any = Depends(get_db),
) -> AnalyzeResponse:
    return await run_analysis(request, settings, db)


@app.post("/api/analyze/start")
async def analyze_start(
    request: AnalyzeRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    job_id = create_job()
    update_job(job_id, stage="queued", percent=1, message="任务已创建，正在启动分析。")
    asyncio.create_task(_run_analysis_job(job_id, request, settings))
    return {"job_id": job_id}


@app.get("/api/analyze/progress/{job_id}")
def analyze_progress(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="analysis job not found")
    return job


@app.get("/api/analyze/result/{job_id}")
def analyze_result(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="analysis job not found")
    result = get_result(job_id)
    if result is None:
        return {"status": job["status"], "ready": False, "error": job.get("error")}
    return {"status": job["status"], "ready": True, "result": jsonable_encoder(result)}


async def _run_analysis_job(job_id: str, request: AnalyzeRequest, settings: Settings) -> None:
    db = SessionLocal() if SessionLocal is not None else None

    def progress(payload: dict[str, Any]) -> None:
        update_job(
            job_id,
            stage=str(payload.get("stage", "running")),
            percent=int(payload.get("percent", 0)),
            message=str(payload.get("message", "")),
            detail=dict(payload.get("detail") or {}),
        )

    try:
        result = await run_analysis(request, settings, db, progress=progress)
        complete_job(job_id, result)
    except Exception as exc:
        fail_job(job_id, str(exc))
    finally:
        if db is not None:
            db.close()


@app.get("/api/companies/search", response_model=list[CompanySearchResult])
async def companies_search(
    q: str,
    settings: Settings = Depends(get_settings),
    db: Any = Depends(get_db),
) -> list[CompanySearchResult]:
    return await search_companies(q, settings, db)


@app.get("/api/llm-key")
def llm_key_status() -> dict[str, bool | str]:
    return {
        "saved": has_deepseek_api_key(),
        "display_value": "••••••••••••" if has_deepseek_api_key() else "",
    }


@app.delete("/api/llm-key")
def llm_key_delete() -> dict[str, bool]:
    delete_deepseek_api_key()
    return {"saved": False}
