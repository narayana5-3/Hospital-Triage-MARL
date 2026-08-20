# Hospital Triage MARL — Command Console

A multi-agent reinforcement learning system for hospital patient triage, wrapped
in a FastAPI backend and a live real-time dashboard. Originally prototyped in a
Colab notebook; this is the productionized version: a real backend, a REST +
WebSocket API, and a browser dashboard that streams triage decisions as they happen.

## What it does

Synthetic patients arrive with vitals (heart rate, BP, SpO2, resp rate, temperature,
consciousness, age, chief complaint). A NEWS2-based clinical scoring function
converts vitals into a severity score. Three independent **DQN agents** — one per
ward (Emergency, General, ICU) — decide where to route and manage patients, guided
by an **LLM triage planner** (Groq/Llama, with a rule-based fallback when no API key
is set) and coordinated through a lightweight **inter-ward message bus** (capacity
alerts, escalation/de-escalation signals). Agents are trained with reward shaping
that penalizes missed critical patients, wrong-ward placement, and deteriorations,
and are rewarded for correct triage and successful transfers.

## Architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│         Frontend             │  REST  │           Backend            │
│  frontend/index.html         │◄──────►│  FastAPI (backend/main.py)   │
│  vanilla JS + canvas          │  WS    │                              │
│  - live ward capacity view    │◄──────►│  Simulation                  │
│  - incoming patient + EKG     │        │   ├─ MultiWardHospitalEnv    │
│  - message bus / event log    │        │   ├─ 3x WardAgent (DQN)      │
│  - training reward chart      │        │   ├─ LLMCoordinatorPlanner   │
└───────────────────────────────┘        │   ├─ MessageBus              │
                                          │   └─ HospitalCoordinator     │
                                          └──────────────────────────────┘
```

- `backend/env/` — patient generator (NEWS2 severity) + the hospital environment
  (ward capacities, transfer logic, per-timestep severity dynamics)
- `backend/agents/` — `WardAgent`: a DQN (128→64→32) per ward with replay buffer,
  target network, epsilon-greedy exploration
- `backend/coordinator/` — `MessageBus` (inter-ward signals) and
  `HospitalCoordinator` (runs each timestep: generate patient → post ward messages
  → LLM plan → agent decisions → environment update → reward → train → audit log)
- `backend/planner/` — `LLMCoordinatorPlanner`: calls Groq's API for a triage
  recommendation, encoded as a 4-float hint fed into every agent's observation;
  falls back to a deterministic rule-based planner if no `GROQ_API_KEY` is set
- `backend/main.py` — FastAPI app exposing REST endpoints and a `/ws/run` WebSocket
  that streams a full 24-step episode in real time
- `backend/train.py` — standalone headless training script
- `frontend/index.html` — single-file dashboard
