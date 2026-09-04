"""Local HTTP API used by Unity to run MatPlotAgent jobs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "local_run.py"
WORKSPACE = ROOT / "workspace" / "api"
UPLOADS = WORKSPACE / "uploads"
JOBS = WORKSPACE / "jobs"

app = FastAPI(title="MatPlotAgent Local API", version="1.0.0")
_jobs: dict[str, dict[str, object]] = {}
_lock = threading.Lock()
_max_concurrent_generations = max(
    1, int(os.getenv("MATPLOT_MAX_CONCURRENT", "9"))
)
_generation_slots = threading.BoundedSemaphore(
    _max_concurrent_generations
)


def _job(job_id: str) -> dict[str, object]:
    with _lock:
        state = _jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return dict(state)


def _update(job_id: str, **values: object) -> None:
    with _lock:
        _jobs[job_id].update(values)


def _write_contract_metadata(
    chart_result: Path,
    grid_contract: Path,
    output_filename: str,
) -> None:
    """Write deterministic provenance from the validated S4D contract.

    MatPlotAgent remains responsible for producing and executing the chart.  The
    required metadata below contains no inferred analysis: every value is copied
    from the validated S4D Grid Contract that governed generation.  Generated
    code may omit a field, use a spelling alias, or accidentally emit an empty
    unit.  None of those formatting mistakes should discard an otherwise valid
    image, and generated code must never be authoritative for data provenance.
    """
    contract = json.loads(grid_contract.read_text(encoding="utf-8"))
    encoding = contract["encoding"]
    spatial = contract["spatialGrid"]
    layout = contract["layout"]
    rows = contract["grid"]["rows"]
    columns = contract["grid"]["columns"]
    cell_order = [
        f"{column['id']}__{row['id']}"
        for row in rows
        for column in columns
    ]
    metadata: dict[str, object] = {}
    if chart_result.is_file():
        try:
            loaded = json.loads(chart_result.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    # Canonical fields always win over model-authored values.
    metadata.update({
        "cellOrder": cell_order,
        "rowOrder": [row["id"] for row in rows],
        "columnOrder": [column["id"] for column in columns],
        "colorMap": encoding.get("colorMap"),
        "minimum": encoding.get("minimum"),
        "maximum": encoding.get("maximum"),
        "unit": encoding.get("unit"),
        "figureWidth": layout.get("figureWidthInches"),
        "figureHeight": layout.get("figureHeightInches"),
        "spatialWidth": spatial.get("width"),
        "spatialHeight": spatial.get("height"),
        "output": output_filename,
        "metadataSource": "validated_grid_contract",
    })
    # Remove known aliases so downstream clients see one stable schema.
    for alias in ("usedCellOrder", "colormap", "outputFilename"):
        metadata.pop(alias, None)
    chart_result.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run(job_id: str, prompt: str, data_path: Path, job_dir: Path) -> None:
    try:
        _update(job_id, status="running", stage="generating initial plot", progress=0.12)
        grid_contract = job_dir / "grid_contract.json"
        command = [
            sys.executable,
            str(RUNNER),
            "--prompt",
            prompt,
            "--data",
            str(data_path),
            "--workspace",
            str(job_dir),
            "--output",
            "final.png",
        ]
        # Grid jobs already carry a strict machine-readable visual contract.
        # One MatPlotAgent generation (plus execution repair when needed) is the
        # production path; generic vision refinement adds latency and can drift
        # from the shared scale/cell-order contract.
        if grid_contract.is_file():
            command.append("--no-visual-refine")
        _update(job_id, stage="waiting for generation slot", progress=0.08)
        with _generation_slots:
            _update(job_id, stage="generating cell panel", progress=0.12)
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=420,
            )
        (job_dir / "api_server.log").write_text(
            (result.stdout or "") + (result.stderr or ""),
            encoding="utf-8",
        )
        final_image = job_dir / "final.png"
        if result.returncode != 0 or not final_image.is_file():
            message = result.stderr or result.stdout or "MatPlotAgent produced no final image."
            raise RuntimeError(message.strip()[-5000:])

        if grid_contract.is_file():
            chart_result = job_dir / "chart_result.json"
            _write_contract_metadata(
                chart_result, grid_contract, final_image.name
            )
            contract = json.loads(grid_contract.read_text(encoding="utf-8"))
            result_metadata = json.loads(chart_result.read_text(encoding="utf-8"))
            expected_encoding = contract["encoding"]
            for field in ("minimum", "maximum", "unit"):
                if result_metadata.get(field) != expected_encoding.get(field):
                    raise RuntimeError(
                        f"S4D chart_result.json has incorrect {field}: "
                        f"expected {expected_encoding.get(field)!r}, "
                        f"found {result_metadata.get(field)!r}"
                    )
            expected_spatial = contract["spatialGrid"]
            for result_field, contract_field in (
                ("spatialWidth", "width"),
                ("spatialHeight", "height"),
            ):
                if result_metadata.get(result_field) != expected_spatial.get(contract_field):
                    raise RuntimeError(
                        f"S4D chart_result.json has incorrect {result_field}: "
                        f"expected {expected_spatial.get(contract_field)!r}, "
                        f"found {result_metadata.get(result_field)!r}"
                    )
            expected_layout = contract["layout"]
            for result_field, contract_field in (
                ("figureWidth", "figureWidthInches"),
                ("figureHeight", "figureHeightInches"),
            ):
                if result_metadata.get(result_field) != expected_layout.get(contract_field):
                    raise RuntimeError(
                        f"S4D chart_result.json has incorrect {result_field}: "
                        f"expected {expected_layout.get(contract_field)!r}, "
                        f"found {result_metadata.get(result_field)!r}"
                    )

        code_file = next(
            (
                path
                for name in ("generated_refined.py", "generated_repair.py", "generated_initial.py")
                if (path := job_dir / name).is_file()
            ),
            None,
        )
        _update(
            job_id,
            status="completed",
            stage="final artifact ready",
            progress=1.0,
            comparison_status="completed",
            review_skipped=not (job_dir / "refined_candidate.png").is_file(),
            rollback_used=(job_dir / "selected_version.txt").is_file()
            and (job_dir / "selected_version.txt").read_text(encoding="utf-8").strip() == "initial",
            image_url=f"/jobs/{job_id}/image",
            code_url=f"/jobs/{job_id}/code" if code_file else "",
            log_url=f"/jobs/{job_id}/log",
        )
    except Exception as exc:
        _update(
            job_id,
            status="failed",
            stage="failed",
            progress=1.0,
            error=str(exc),
        )


@app.get("/health")
def health() -> dict[str, object]:
    provider = os.getenv("MATPLOT_PROVIDER", "").lower()
    use_qwen = provider == "qwen" or (
        not provider and bool(os.getenv("DASHSCOPE_API_KEY"))
    )
    return {
        "status": "ok",
        "runner_available": RUNNER.is_file(),
        "workspaceRoot": str(ROOT),
        "workspace": str(WORKSPACE),
        "provider": "qwen" if use_qwen else "openai",
        "providerConfigured": bool(
            os.getenv("DASHSCOPE_API_KEY") if use_qwen
            else os.getenv("OPENAI_API_KEY")
        ),
        "max_concurrent_generations": _max_concurrent_generations,
        "code_model": os.getenv(
            "MATPLOT_CODE_MODEL",
            os.getenv(
                "MATPLOT_MODEL",
                "qwen-flash" if use_qwen else "gpt-4.1-mini",
            ),
        ),
    }


@app.post("/jobs")
async def create_job(
    prompt: str = Form(...),
    data: UploadFile = File(...),
    contract: Optional[UploadFile] = File(None),
) -> dict[str, str]:
    if not RUNNER.is_file():
        raise HTTPException(status_code=503, detail=f"Missing runner: {RUNNER}")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    job_id = uuid.uuid4().hex
    suffix = Path(data.filename or "data.csv").suffix or ".csv"
    upload_path = UPLOADS / f"{job_id}{suffix}"
    job_dir = JOBS / job_id
    UPLOADS.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=False)
    upload_path.write_bytes(await data.read())
    if contract is not None:
        contract_path = job_dir / "grid_contract.json"
        contract_path.write_bytes(await contract.read())
        try:
            json.loads(contract_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail="Invalid Grid contract JSON") from exc

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.04,
            "comparison_status": "",
            "review_skipped": False,
            "rollback_used": False,
            "error": "",
            "image_url": "",
            "code_url": "",
            "log_url": "",
        }
    threading.Thread(
        target=_run,
        args=(job_id, prompt.strip(), upload_path, job_dir),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/jobs/{job_id}",
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    return _job(job_id)


def _artifact(job_id: str, filename: str, media_type: str) -> FileResponse:
    _job(job_id)
    path = JOBS / job_id / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact is unavailable: {filename}")
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/jobs/{job_id}/image")
def final_image(job_id: str) -> FileResponse:
    return _artifact(job_id, "final.png", "image/png")


@app.get("/jobs/{job_id}/initial-image")
def initial_image(job_id: str) -> FileResponse:
    return _artifact(job_id, "initial.png", "image/png")


@app.get("/jobs/{job_id}/candidate-image")
def candidate_image(job_id: str) -> FileResponse:
    return _artifact(job_id, "refined_candidate.png", "image/png")


@app.get("/jobs/{job_id}/code", response_class=PlainTextResponse)
def generated_code(job_id: str) -> str:
    _job(job_id)
    for name in ("generated_refined.py", "generated_repair.py", "generated_initial.py"):
        path = JOBS / job_id / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    raise HTTPException(status_code=404, detail="Generated code is unavailable")


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def generated_log(job_id: str) -> str:
    _job(job_id)
    parts: list[str] = []
    for path in sorted((JOBS / job_id).glob("*.log")):
        parts.append(f"--- {path.name} ---\n{path.read_text(encoding='utf-8', errors='replace')}")
    if not parts:
        raise HTTPException(status_code=404, detail="Job log is unavailable")
    return "\n\n".join(parts)


@app.get("/jobs/{job_id}/metadata")
def chart_metadata(job_id: str) -> FileResponse:
    return _artifact(job_id, "chart_result.json", "application/json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("MATPLOT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("MATPLOT_API_PORT", "8010")),
    )
