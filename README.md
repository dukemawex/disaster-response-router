# disaster-response-router

`disaster-response-router` classifies incoming disaster messages, prioritizes them, and simulates queue-based routing latency.

## Features

- **Message triage classifier** for `high`, `medium`, and `low` urgency.
- **Priority queue simulation** with configurable responders and service time.
- **Metrics**:
  - macro **F1 score**
  - **mean routing latency** (minutes)
- **Tavily integration** (optional): fetch disaster triage principles from the web.
- **Gemini integration** (optional): draft operational guidelines from those principles.

If API keys are not provided, the router uses robust local defaults.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Environment variables (optional)

- `TAVILY_API_KEY` — enables live principle retrieval via Tavily.
- `GEMINI_API_KEY` — enables guideline generation with Gemini.

## Run tests

```bash
pytest -q
```
