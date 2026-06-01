from __future__ import annotations

import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"
    progress: int = 0
    stage: str = "queued"
    logs: list[str] = field(default_factory=list)
    partial_text: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def log(self, message: str) -> None:
        self.updated_at = time.time()
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {message}")

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = round(self.created_at, 3)
        data["updated_at"] = round(self.updated_at, 3)
        return data


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, kind: str) -> Job:
        job = Job(id=job_id, kind=kind)
        job.log("任务已创建")
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> Optional[dict[str, Any]]:
        job = self.get(job_id)
        return job.snapshot() if job else None

    def start(self, job: Job, target: Callable[[Job], None]) -> None:
        def runner() -> None:
            try:
                job.status = "running"
                job.stage = "running"
                job.progress = max(job.progress, 5)
                job.log("后台任务开始执行")
                target(job)
                if job.status != "failed":
                    job.status = "succeeded"
                    job.stage = "done"
                    job.progress = 100
                    job.log("任务完成")
            except Exception as exc:
                job.status = "failed"
                job.stage = "failed"
                job.error = str(exc)
                job.log(f"任务失败：{exc}")
                tb_path = Path("data/outputs") / f"{job.id}.traceback.log"
                tb_path.parent.mkdir(parents=True, exist_ok=True)
                tb_path.write_text(traceback.format_exc(), encoding="utf-8")
                job.log(f"错误详情已写入 {tb_path}")
            finally:
                job.updated_at = time.time()

        thread = threading.Thread(target=runner, name=f"qwen-asr-job-{job.id}", daemon=True)
        thread.start()
