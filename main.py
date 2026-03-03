from src.disaster_response_router import (
    draft_operational_guidelines,
    evaluate_f1,
    retrieve_triage_principles,
    sample_messages,
    simulate_queue,
)


def main() -> None:
    messages = sample_messages()
    principles = retrieve_triage_principles()
    guidelines = draft_operational_guidelines(principles)

    f1 = evaluate_f1(messages)
    latency = simulate_queue(messages, workers=3)

    print("=== Disaster Response Router ===")
    print("Triage principles:")
    for idx, p in enumerate(principles, start=1):
        print(f"{idx}. {p}")

    print("\nOperational guidelines:")
    print(guidelines)

    print("\nMetrics:")
    print(f"F1: {f1:.3f}")
    print(f"Mean routing latency (minutes): {latency:.2f}")


if __name__ == "__main__":
    main()
