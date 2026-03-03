from __future__ import annotations

import heapq
import json
import os
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from statistics import mean
from typing import Iterable


DEFAULT_TRIAGE_PRINCIPLES = [
    "Life-threatening conditions receive immediate escalation.",
    "Prioritize active hazards (fire, flooding, structural collapse, gas leaks).",
    "Prioritize vulnerable populations (children, elderly, disabled).",
    "Escalate messages with trapped victims, severe injury, or missing persons.",
    "De-prioritize informational updates with no immediate safety risk.",
]


@dataclass(frozen=True)
class Message:
    text: str
    truth: str


def _post_json(url: str, payload: dict, params: dict[str, str] | None = None, timeout: int = 10) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def retrieve_triage_principles() -> list[str]:
    """Fetch triage principles from Tavily if API key exists; else local defaults."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return DEFAULT_TRIAGE_PRINCIPLES

    prompt = (
        "List concise disaster-response triage principles used by emergency operations "
        "centers. Focus on prioritization and routing urgency."
    )
    try:
        payload = _post_json(
            "https://api.tavily.com/search",
            {"api_key": api_key, "query": prompt, "max_results": 5},
            timeout=10,
        )
        snippets = [item.get("content", "") for item in payload.get("results", [])]
        text = "\n".join(snippets)
        principles = [
            sentence.strip(" -•\t")
            for sentence in re.split(r"[\n\r\.]+", text)
            if len(sentence.strip()) > 30
        ]
        return principles[:8] if principles else DEFAULT_TRIAGE_PRINCIPLES
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        return DEFAULT_TRIAGE_PRINCIPLES


def draft_operational_guidelines(principles: Iterable[str]) -> str:
    """Generate routing guidance via Gemini when API key is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    principle_text = "\n".join(f"- {p}" for p in principles)

    if not api_key:
        return (
            "Operational guidelines:\n"
            "1. Escalate high-risk life-safety incidents immediately.\n"
            "2. Route active hazards to fastest available responder queue.\n"
            "3. Mark welfare checks as medium unless severe risk indicators appear.\n"
            "4. Keep informational reports in low-priority backlog.\n"
            f"Reference principles:\n{principle_text}"
        )

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Draft concise operational guidelines for classifying and routing "
                            "disaster-response messages based on these principles:\n"
                            f"{principle_text}"
                        )
                    }
                ]
            }
        ]
    }
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    try:
        payload = _post_json(url, body, params={"key": api_key}, timeout=15)
        return payload["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (urllib.error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        return (
            "Operational guidelines unavailable from Gemini. Using fallback process:\n"
            "- Prioritize life-threatening and active-hazard incidents.\n"
            "- Route non-critical requests after urgent queues are drained."
        )


HIGH_KEYWORDS = {
    "trapped",
    "collapsed",
    "severe",
    "bleeding",
    "unconscious",
    "fire",
    "explosion",
    "flood",
    "gas leak",
    "missing",
    "injured",
}
MEDIUM_KEYWORDS = {
    "power outage",
    "road blocked",
    "needs medicine",
    "elderly",
    "disabled",
    "evacuation",
    "shelter",
    "water needed",
}


def classify_message(text: str) -> str:
    normalized = text.lower()
    high_hits = sum(1 for k in HIGH_KEYWORDS if k in normalized)
    medium_hits = sum(1 for k in MEDIUM_KEYWORDS if k in normalized)

    if high_hits > 0:
        return "high"
    if medium_hits > 0:
        return "medium"
    return "low"


def _f1_for_label(y_true: list[str], y_pred: list[str], label: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p == label)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != label and p == label)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == label and p != label)

    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp == 0:
        return 1.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def evaluate_f1(messages: list[Message]) -> float:
    y_true = [m.truth for m in messages]
    y_pred = [classify_message(m.text) for m in messages]
    labels = ["high", "medium", "low"]
    return mean(_f1_for_label(y_true, y_pred, label) for label in labels)


def simulate_queue(
    messages: list[Message],
    workers: int = 3,
    seed: int = 7,
    service_minutes: tuple[int, int] = (4, 12),
) -> float:
    """Simulate priority routing and return mean latency in minutes."""
    rng = random.Random(seed)
    arrivals = list(range(len(messages)))
    pri_map = {"high": 0, "medium": 1, "low": 2}

    worker_free_at = [0 for _ in range(workers)]
    pq: list[tuple[int, int, Message]] = []
    latencies: list[float] = []

    for t, message in zip(arrivals, messages, strict=True):
        predicted = classify_message(message.text)
        heapq.heappush(pq, (pri_map[predicted], t, message))

        next_worker = min(range(workers), key=lambda i: worker_free_at[i])
        if worker_free_at[next_worker] <= t and pq:
            _pri, enqueue_t, _msg = heapq.heappop(pq)
            start_t = t
            service_t = rng.randint(*service_minutes)
            worker_free_at[next_worker] = start_t + service_t
            latencies.append(float(start_t - enqueue_t))

    while pq:
        next_worker = min(range(workers), key=lambda i: worker_free_at[i])
        _pri, enqueue_t, _msg = heapq.heappop(pq)
        start_t = worker_free_at[next_worker]
        service_t = rng.randint(*service_minutes)
        worker_free_at[next_worker] = start_t + service_t
        latencies.append(float(start_t - enqueue_t))

    return mean(latencies) if latencies else 0.0


def sample_messages() -> list[Message]:
    return [
        Message("Family trapped after building collapsed near river.", "high"),
        Message("Need shelter space for elderly residents.", "medium"),
        Message("Status update: rain slowing in north district.", "low"),
        Message("Gas leak reported; multiple injured people.", "high"),
        Message("Power outage affecting neighborhood clinic.", "medium"),
        Message("Requesting blanket donation drop-off location.", "low"),
        Message("Unconscious person found after flood waters rose.", "high"),
        Message("Road blocked by debris, no injuries reported.", "medium"),
    ]
