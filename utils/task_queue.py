import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from utils.logger import get_logger
from utils.metrics import observe_backtest_duration

logger = get_logger()


class JobStatus:
    def __init__(self):
        self.status: str = "queued"  # queued, running, done, error
        self.progress: float = 0.0
        self.message: str = ""
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.logs: list[str] = []
        self.lock = threading.Lock()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "logs": list(self.logs),
        }


class TaskQueue:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, JobStatus] = {}
        self.lock = threading.Lock()

    def submit(self, func: Callable[..., dict], *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex
        status = JobStatus()
        with self.lock:
            self.jobs[job_id] = status

        def progress_cb(pct: float, msg: str = ""):
            with status.lock:
                status.progress = max(0.0, min(100.0, pct))
                status.message = msg
                status.logs.append(f"{time.strftime('%H:%M:%S')} {pct:.1f}% {msg}")

        def wrapped():
            start = time.perf_counter()
            with status.lock:
                status.status = "running"
                status.started_at = time.time()
            try:
                result = func(progress_cb=progress_cb, *args, **kwargs)
                with status.lock:
                    status.status = "done"
                    status.progress = 100.0
                    status.result = result
                    status.ended_at = time.time()
                observe_backtest_duration(time.perf_counter() - start, status="ok")
            except Exception as e:
                logger.exception("Job %s failed: %s", job_id, str(e))
                with status.lock:
                    status.status = "error"
                    status.error = str(e)
                    status.ended_at = time.time()
                observe_backtest_duration(time.perf_counter() - start, status="error")

        self.executor.submit(wrapped)
        return job_id

    def get(self, job_id: str) -> Optional[JobStatus]:
        with self.lock:
            return self.jobs.get(job_id)


# Singleton for app
GLOBAL_TASK_QUEUE = TaskQueue()
