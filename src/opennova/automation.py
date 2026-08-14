"""OpenNova中的自动化任务模块，集中定义相关数据结构、边界适配和实现逻辑。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from opennova.runtime.cancellation import CancellationToken


def compute_retry_delay(
    attempt: int, base_seconds: float = 1.0, max_seconds: float = 60.0
) -> float:
    """计算 `retry_delay` 对应的数据，并按照当前组件的约定返回结果。

    参数：
        attempt: 本次操作使用的`attempt`。
        base_seconds: 可选的`base_seconds`。
        max_seconds: 可选的`max_seconds`。

    返回：
        `float` 类型的处理结果。
    """
    return min(max_seconds, base_seconds * (2.0 ** max(0, attempt)))


@dataclass
class ScheduledTask:
    """保存定时任务所需的结构化数据，主要包含 `id`、`name`、`prompt`、`next_run_at`、`interval_seconds`、`enabled`
    字段，便于在组件之间传递或持久化。
    """

    id: str
    name: str
    prompt: str
    next_run_at: float
    interval_seconds: float | None = None
    enabled: bool = True


@dataclass
class ScheduledRun:
    """保存定时任务运行所需的结构化数据，主要包含 `task_id`、`task_name`、`ran_at`、`success`、`output`、`error`
    字段，便于在组件之间传递或持久化。
    """

    task_id: str
    task_name: str
    ran_at: float
    success: bool
    output: str = ""
    error: str | None = None


class LocalAutomationScheduler:
    """封装`LocalAutomationScheduler`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, storage_path: str | Path, clock: Callable[[], float] = time.time):
        self.storage_path = Path(storage_path)
        self.clock = clock
        self.tasks: dict[str, ScheduledTask] = {}
        self.history: list[ScheduledRun] = []
        self._load()

    def schedule_once(self, name: str, prompt: str, run_at: float) -> str:
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            prompt=prompt,
            next_run_at=run_at,
        )
        self.tasks[task.id] = task
        self.save()
        return task.id

    def schedule_interval(
        self,
        name: str,
        prompt: str,
        interval_seconds: float,
        start_at: float | None = None,
    ) -> str:
        now = self.clock()
        task = ScheduledTask(
            id=str(uuid.uuid4()),
            name=name,
            prompt=prompt,
            next_run_at=start_at if start_at is not None else now,
            interval_seconds=interval_seconds,
        )
        self.tasks[task.id] = task
        self.save()
        return task.id

    def get(self, task_id: str) -> ScheduledTask:
        return self.tasks[task_id]

    def list_tasks(self) -> list[ScheduledTask]:
        return sorted(self.tasks.values(), key=lambda task: task.next_run_at)

    def pause(self, task_id: str) -> None:
        self.tasks[task_id].enabled = False
        self.save()

    def resume(self, task_id: str) -> None:
        self.tasks[task_id].enabled = True
        self.save()

    def delete(self, task_id: str) -> None:
        self.tasks.pop(task_id)
        self.save()

    def due_tasks(self) -> list[ScheduledTask]:
        now = self.clock()
        return [task for task in self.tasks.values() if task.enabled and task.next_run_at <= now]

    def run_now(
        self,
        task_id: str,
        runner: Callable[[ScheduledTask], object],
        cancellation_token: CancellationToken | None = None,
    ) -> ScheduledRun:
        task = self.tasks[task_id]
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        run = self._run_task(task, runner)
        if task.interval_seconds and task.enabled:
            task.next_run_at = self.clock() + task.interval_seconds
        else:
            task.enabled = False
        self.save()
        return run

    def run_due(
        self,
        runner: Callable[[ScheduledTask], object],
        cancellation_token: CancellationToken | None = None,
    ) -> list[str]:
        ran: list[str] = []
        now = self.clock()
        for task in self.due_tasks():
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            self._run_task(task, runner)
            ran.append(task.id)
            if task.interval_seconds:
                task.next_run_at = now + task.interval_seconds
            else:
                task.enabled = False
        if ran:
            self.save()
        return ran

    def _run_task(
        self, task: ScheduledTask, runner: Callable[[ScheduledTask], object]
    ) -> ScheduledRun:
        try:
            output = runner(task)
            run = ScheduledRun(
                task_id=task.id,
                task_name=task.name,
                ran_at=self.clock(),
                success=True,
                output="" if output is None else str(output),
            )
        except Exception as exc:
            run = ScheduledRun(
                task_id=task.id,
                task_name=task.name,
                ran_at=self.clock(),
                success=False,
                error=str(exc),
            )
        self.history.append(run)
        return run

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tasks": [asdict(task) for task in self.tasks.values()],
            "history": [asdict(run) for run in self.history],
        }
        self.storage_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for task_data in payload.get("tasks", []):
            task = ScheduledTask(**task_data)
            self.tasks[task.id] = task
        for run_data in payload.get("history", []):
            self.history.append(ScheduledRun(**run_data))


class AutomationArchive:
    """封装`AutomationArchive`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, archive_dir: str | Path):
        self.archive_dir = Path(archive_dir)
        self.path = self.archive_dir / "automation-events.jsonl"

    def append_event(self, event: dict[str, object]) -> Path:
        """追加事件，并按照当前组件的约定返回结果。

        参数：
            event: 需要处理或发布的运行时事件。

        返回：
            `Path` 类型的处理结果。

        说明：
            该操作会访问本地文件系统，路径校验和原子写入约束由所在组件负责。
        """
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return self.path

    def read_events(self) -> list[dict[str, object]]:
        """读取事件，并按照当前组件的约定返回结果。

        返回：
            按调用约定排序的结果列表。
        """
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def summary(self) -> dict[str, object]:
        """处理摘要，并按照当前组件的约定返回结果。

        返回：
            供后续逻辑或序列化使用的结构化字典。
        """
        events = self.read_events()
        return {
            "total": len(events),
            "failed": sum(1 for event in events if event.get("success") is False),
            "last_event": events[-1] if events else None,
        }


def daemon_status(
    daemon: LocalAutomationDaemon,
    archive: AutomationArchive | None = None,
) -> dict[str, object]:
    """读取并返回 `daemon_status` 所表示的数据或流程，并遵守当前模块定义的边界与状态约束。

    参数：
        daemon: 本次操作使用的守护进程。
        archive: 可选的`archive`。

    返回：
        供后续逻辑或序列化使用的结构化字典。
    """
    status: dict[str, object] = {
        "running": daemon.running,
        "last_events_count": len(daemon.last_events),
        "last_events": daemon.last_events,
    }
    if archive:
        status["archive"] = archive.summary()
    return status


class LocalAutomationMonitor:
    """封装`LocalAutomationMonitor`相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, scheduler: LocalAutomationScheduler):
        self.scheduler = scheduler

    def tick(
        self,
        runner: Callable[[ScheduledTask], object],
        cancellation_token: CancellationToken | None = None,
    ) -> list[dict[str, object]]:
        """启动或推进 `tick` 所表示的数据或流程，并遵守`LocalAutomationMonitor`定义的边界与状态约束。

        参数：
            runner: 本次操作使用的`runner`。
            cancellation_token: 由 TUI、SDK 和工具共享的取消信号。

        返回：
            按调用约定排序的结果列表。
        """
        due = self.scheduler.due_tasks()
        events: list[dict[str, object]] = []
        for task in due:
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            run = self.scheduler.run_now(task.id, runner, cancellation_token)
            events.append(
                {
                    "type": "automation_run",
                    "task_id": task.id,
                    "task_name": task.name,
                    "success": run.success,
                    "output": run.output,
                    "error": run.error,
                    "ran_at": run.ran_at,
                }
            )
        return events


class LocalAutomationDaemon:
    """封装本地自动化任务守护进程相关的状态和操作，使调用方通过稳定接口使用该能力。"""

    def __init__(self, scheduler: LocalAutomationScheduler):
        self.monitor = LocalAutomationMonitor(scheduler)
        self.running = False
        self.last_events: list[dict[str, object]] = []

    def start(self) -> None:
        """处理启动，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.running = True

    def stop(self) -> None:
        """处理停止，并按照当前组件的约定返回结果。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        self.running = False

    def run_once(
        self,
        runner: Callable[[ScheduledTask], object],
        cancellation_token: CancellationToken | None = None,
    ) -> list[dict[str, object]]:
        """运行`run_once`流程，并统一处理完成、失败和取消。

        参数：
            runner: 本次操作使用的`runner`。
            cancellation_token: 由 TUI、SDK 和工具共享的取消信号。

        返回：
            按调用约定排序的结果列表。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if not self.running:
            return []
        self.last_events = self.monitor.tick(runner, cancellation_token)
        return self.last_events

    def run_until_idle(
        self,
        runner: Callable[[ScheduledTask], object],
        max_ticks: int = 10,
        cancellation_token: CancellationToken | None = None,
    ) -> list[dict[str, object]]:
        """运行直到空闲流程，并统一处理完成、失败和取消。

        参数：
            runner: 本次操作使用的`runner`。
            max_ticks: 可选的`max_ticks`。
            cancellation_token: 由 TUI、SDK 和工具共享的取消信号。

        返回：
            按调用约定排序的结果列表。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if not self.running:
            return []

        all_events: list[dict[str, object]] = []
        for _ in range(max_ticks):
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
            events = self.monitor.tick(runner, cancellation_token)
            if not events:
                break
            all_events.extend(events)
        self.last_events = all_events
        return all_events

    def run_with_retry(
        self,
        runner: Callable[[ScheduledTask], object],
        max_retries: int = 1,
        archive_callback: Callable[[dict[str, object]], object] | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[dict[str, object]]:
        """运行带重试的重试流程，并统一处理完成、失败和取消。

        参数：
            runner: 本次操作使用的`runner`。
            max_retries: 可选的`max_retries`。
            archive_callback: 可选的`archive_callback`。
            cancellation_token: 由 TUI、SDK 和工具共享的取消信号。

        返回：
            按调用约定排序的结果列表。

        说明：
            执行过程中会更新当前实例维护的状态。
        """
        if not self.running:
            return []

        events: list[dict[str, object]] = []
        for task in list(self.monitor.scheduler.due_tasks()):
            attempts = 0
            while True:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                run = self.monitor.scheduler.run_now(task.id, runner, cancellation_token)
                if run.success:
                    event = self._event_from_run(run, "automation_run")
                    events.append(event)
                    break
                if attempts >= max_retries:
                    event = self._event_from_run(run, "automation_run")
                    events.append(event)
                    break
                attempts += 1
                task.enabled = True
                task.next_run_at = self.monitor.scheduler.clock()
                self.monitor.scheduler.tasks[task.id] = task
                retry_event = self._event_from_run(run, "automation_retry")
                retry_event["attempt"] = attempts
                events.append(retry_event)
        self.last_events = events
        if archive_callback:
            for event in events:
                archive_callback(event)
        return events

    def _event_from_run(self, run: ScheduledRun, event_type: str) -> dict[str, object]:
        return {
            "type": event_type,
            "task_id": run.task_id,
            "task_name": run.task_name,
            "success": run.success,
            "output": run.output,
            "error": run.error,
            "ran_at": run.ran_at,
        }
