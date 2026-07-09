from tandem_rlvr.tasks.splits import build_split_benchmark


def test_split_metadata_on_generated_tasks() -> None:
    tasks = build_split_benchmark(2, splits=["id_eval"], seed=1, task_families=["arithmetic", "list"])

    assert len(tasks) == 2
    for task in tasks:
        assert task.metadata["split"] == "id_eval"
        assert task.metadata["distribution"] == "id"
        assert task.metadata["ood_type"]
        assert task.metadata["task_family"] in {"arithmetic", "list"}


def test_id_ood_stress_arithmetic_generation() -> None:
    tasks = build_split_benchmark(3, splits=["id_eval", "ood_eval", "stress_eval"], seed=2, task_families=["arithmetic"])

    by_split = {task.metadata["split"]: task for task in tasks}

    assert by_split["id_eval"].difficulty == "easy"
    assert by_split["ood_eval"].metadata["distribution"] == "ood"
    assert by_split["stress_eval"].metadata["distribution"] == "stress"


def test_id_ood_stress_list_generation() -> None:
    tasks = build_split_benchmark(3, splits=["id_eval", "ood_eval", "stress_eval"], seed=3, task_families=["list"])

    assert {task.metadata["task_family"] for task in tasks} == {"list"}
    assert {task.metadata["split"] for task in tasks} == {"id_eval", "ood_eval", "stress_eval"}
    assert all(task.metadata["answer_type"] == "list_int" for task in tasks)


def test_id_ood_stress_logic_generation() -> None:
    tasks = build_split_benchmark(3, splits=["id_eval", "ood_eval", "stress_eval"], seed=4, task_families=["logic"])

    assert {task.metadata["task_family"] for task in tasks} == {"logic"}
    assert all(task.metadata["answer_type"] in {"bool", "text"} for task in tasks)


def test_id_ood_stress_code_generation() -> None:
    tasks = build_split_benchmark(3, splits=["id_eval", "ood_eval", "stress_eval"], seed=5, task_families=["code"])

    assert {task.metadata["task_family"] for task in tasks} == {"code"}
    assert all(task.task_type.startswith("code_trace_") for task in tasks)
    assert all(task.metadata["answer_type"] == "int" for task in tasks)
