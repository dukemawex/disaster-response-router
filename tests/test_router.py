from src.disaster_response_router import (
    Message,
    classify_message,
    evaluate_f1,
    sample_messages,
    simulate_queue,
)


def test_classify_message() -> None:
    assert classify_message("People trapped in fire zone") == "high"
    assert classify_message("Need shelter and water needed in district 7") == "medium"
    assert classify_message("General weather bulletin update") == "low"


def test_f1_is_reasonable_on_sample() -> None:
    score = evaluate_f1(sample_messages())
    assert 0.7 <= score <= 1.0


def test_queue_latency_non_negative() -> None:
    data = [
        Message("fire reported downtown", "high"),
        Message("road blocked near bridge", "medium"),
        Message("status update", "low"),
    ]
    latency = simulate_queue(data, workers=1, seed=1, service_minutes=(2, 2))
    assert latency >= 0
