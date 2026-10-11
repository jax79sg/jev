"""ClearML wiring shared by every training/eval run in this project."""
from clearml import Task

PROJECT_NAME = "jev-transaction-decision-model"


def init_task(task_name: str, config: dict, task_type=Task.TaskTypes.training, tags=None) -> Task:
    """Open a ClearML Task and connect the full run config before any work starts."""
    task = Task.init(project_name=PROJECT_NAME, task_name=task_name, task_type=task_type, tags=tags or [])
    task.connect(config, name="config")
    return task
