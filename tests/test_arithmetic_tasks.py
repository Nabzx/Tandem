from tandem_rlvr.tasks.arithmetic import ArithmeticTaskGenerator


def test_generator_creates_requested_number_of_tasks() -> None:
    generator = ArithmeticTaskGenerator(seed=1, difficulty="easy")

    tasks = generator.generate(5)

    assert len(tasks) == 5
    assert [task.task_id for task in tasks] == [
        "arith-000001",
        "arith-000002",
        "arith-000003",
        "arith-000004",
        "arith-000005",
    ]


def test_easy_arithmetic_operands_are_small() -> None:
    generator = ArithmeticTaskGenerator(seed=2, difficulty="easy")

    task = generator.generate_one()

    assert 0 <= task.metadata["left"] <= 9
    assert 0 <= task.metadata["right"] <= 9
    assert task.difficulty == "easy"


def test_addition_answer_is_exact() -> None:
    generator = ArithmeticTaskGenerator(seed=3, difficulty="medium", task_types=["addition"])

    task = generator.generate_one()

    assert int(task.answer) == task.metadata["left"] + task.metadata["right"]
    assert task.task_type == "addition"
