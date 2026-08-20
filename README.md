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
- `frontend/index.html` — single-file dashboard (no build step, no framework)

## Quickstart

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt

# The repo ships with checkpoints already trained for ~400 episodes,
# so you can skip straight to running the server. To retrain from scratch:
#   python train.py --episodes 500

uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** — the FastAPI app serves the dashboard
directly, so there's no separate frontend server to run.

Click **Run Episode** to stream a full 24-timestep episode live over WebSocket,
or **Step** to advance one patient at a time and inspect each decision.

### Optional: enable the real LLM planner

By default there's no `GROQ_API_KEY`, so the system uses a deterministic
rule-based fallback planner (this is by design — the notebook's fallback logic
is already sound and keeps the demo dependency-free). To use the actual Groq
LLM instead:

```bash
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

Get a free key at [console.groq.com](https://console.groq.com).

## API reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server + device status |
| `/api/state` | GET | Current episode snapshot |
| `/api/reset` | POST | Start a new episode |
| `/api/step` | POST | Advance one patient/timestep |
| `/api/train` | POST | Run additional training episodes `{episodes: N}` |
| `/api/reward_history` | GET | Training reward curve (from `train.py` runs) |
| `/ws/run` | WebSocket | Streams a full episode step-by-step (`{delay_ms}` on connect) |

## Retraining

```bash
python train.py --episodes 500          # fresh run
python train.py --episodes 1000 --resume  # continue from saved checkpoints
```

Checkpoints save to `backend/models/`, which the API loads automatically on
startup. Retraining is fast — CPU-only, ~400 episodes in under a minute, since
each episode is just 24 timesteps through small (128-unit) networks.

## Deploying it for your portfolio

For a link you can put on a resume:

- **Backend**: any container host with a persistent disk works (Render, Railway,
  Fly.io). Point the start command at `uvicorn main:app --host 0.0.0.0 --port $PORT`.
  Since checkpoints are committed to the repo, no training step is needed at deploy time.
- **Frontend**: already served by FastAPI's `StaticFiles` mount, so a single
  deployed service is enough — no separate static host needed.
- If you outgrow the single-episode-at-a-time design (e.g. multiple visitors),
  swap the global `Simulation` singleton in `main.py` for a per-session store
  keyed by a session ID.

## Notes on what changed from the notebook

- Removed the Colab/Google Drive mounting and the unused `gym.Env` inheritance
  (the environment never used Gym's spaces API, so the dependency was dead weight).
- Split the monolithic notebook cells into importable modules — the core
  simulation logic (severity scoring, environment dynamics, DQN agent, message
  bus, planner, coordinator) is otherwise unchanged from what you validated in
  the notebook.
- Added `wards_summary()` / `to_dict()` serialization methods so the environment
  state can be sent over the API as JSON.
- `HospitalCoordinator.step()` now returns the full per-step payload (patient,
  ward states, messages, plan, confidence) needed to drive the live dashboard.
