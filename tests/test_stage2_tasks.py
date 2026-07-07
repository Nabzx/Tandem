from tandem_rlvr.tasks import CodeTracingTaskGenerator, ListTransformationTaskGenerator, LogicTaskGenerator
from tandem_rlvr.tasks.verifiers import verify_final_answer


def test_list_transformation_generator_creates_structured_answers() -> None:
    generator = ListTransformationTaskGenerator(seed=1, task_types=["list_sort"], difficulty="easy")

    task = generator.generate_one()

    assert task.task_type == "list_sort"
    assert task.metadata["answer_type"] == "list_int"
    assert verify_final_answer(task, task.answer)


def test_logic_generator_creates_verifiable_boolean_task() -> None:
    generator = LogicTaskGenerator(seed=2, task_types=["logic_syllogism"], difficulty="easy")

    task = generator.generate_one()

    assert task.task_type == "logic_syllogism"
    assert task.metadata["answer_type"] == "bool"
    assert verify_final_answer(task, "yes")


def test_code_tracing_generator_uses_known_template_answer() -> None:
    generator = CodeTracingTaskGenerator(seed=3, task_types=["code_trace_loop_sum"], difficulty="easy")

    task = generator.generate_one()

    assert task.task_type == "code_trace_loop_sum"
    assert task.metadata["answer_type"] == "int"
    assert int(task.answer) == sum(task.metadata["items"])
