import asyncio
import logging
import os

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env.hospital_env import MultiWardHospitalEnv
from env.patient_generator import SyntheticPatientGenerator
from agents.ward_agent import WardAgent
from coordinator.message_bus import MessageBus
from coordinator.hospital_coordinator import HospitalCoordinator
from planner.llm_planner import LLMCoordinatorPlanner

logging.basicConfig(level=logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MEMORY_DIR = os.path.join(BASE_DIR, 'memory')
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')

app = FastAPI(title="Hospital Triage MARL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Simulation:
    """
    Holds the live simulation state: environment, trained ward agents,
    LLM planner, message bus, and coordinator. One shared instance
    is used for the whole demo (single active episode at a time).
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset_components()
        self.episode = 0
        self.step_idx = 0
        self.done = True
        self.total_reward = 0.0
        self.history = []          # step-by-step log for the current episode
        self.load_checkpoints()

    def reset_components(self):
        self.env = MultiWardHospitalEnv()
        self.patient_gen = SyntheticPatientGenerator()
        self.ward_agents = {
            name: WardAgent(ward_name=name, device=self.device,
                             model_dir=MODEL_DIR, memory_dir=MEMORY_DIR)
            for name in ['Emergency', 'General', 'ICU']
        }
        self.planner = LLMCoordinatorPlanner(plan_interval=4)
        self.bus = MessageBus(log_dir=LOG_DIR)
        self.coordinator = HospitalCoordinator(
            env=self.env, patient_gen=self.patient_gen,
            ward_agents=self.ward_agents, planner=self.planner,
            message_bus=self.bus, log_dir=LOG_DIR, model_dir=MODEL_DIR,
        )

    def load_checkpoints(self):
        try:
            self.coordinator.load_all()
            # Demo mode: greedy, trained agents (no exploration noise)
            for agent in self.ward_agents.values():
                agent.epsilon = 0.0
            self.trained = True
        except Exception as e:
            logging.warning(f'Could not load checkpoints: {e}')
            self.trained = False

    def reset_episode(self):
        self.env.reset()
        self.episode += 1
        self.step_idx = 0
        self.done = False
        self.total_reward = 0.0
        self.history = []
        return self.snapshot()

    def take_step(self, train=False):
        if self.done:
            return None
        reward, done, info = self.coordinator.step(
            self.episode, self.step_idx, train=train, verbose=False)
        self.total_reward += reward
        self.step_idx += 1
        self.done = done
        entry = {
            'step': self.step_idx,
            'reward': round(float(reward), 2),
            'total_reward': round(float(self.total_reward), 2),
            'done': done,
            'patient': _serialize_patient(info['patient']),
            'em_action': info['em_action'],
            'em_action_name': info['em_action_name'],
            'em_confidence': info['em_confidence'],
            'outcome': info['outcome'],
            'plan': info['plan'],
            'messages': info['messages'],
            'ward_states': info['ward_states'],
        }
        self.history.append(entry)
        return entry

    def snapshot(self):
        return {
            'episode': self.episode,
            'step': self.step_idx,
            'done': self.done,
            'total_reward': round(float(self.total_reward), 2),
            'trained': self.trained,
            'ward_states': self.env.wards_summary(),
            'llm_backend': self.planner.stats,
        }


def _serialize_patient(p):
    return {
        'id': p['id'],
        'severity': p['severity'],
        'severity_band': p['severity_band'],
        'vitals': p['vitals'],
    }


sim = Simulation()


@app.get("/api/health")
def health():
    return {"status": "ok", "trained": sim.trained, "device": str(sim.device)}


@app.get("/api/state")
def get_state():
    return sim.snapshot()


@app.post("/api/reset")
def reset():
    return sim.reset_episode()


@app.post("/api/step")
def step():
    if sim.done:
        return {"error": "Episode finished. Call /api/reset first.", "done": True}
    entry = sim.take_step(train=False)
    return entry


@app.get("/api/reward_history")
def reward_history():
    """Training reward curve, from the pretraining run (models/reward_history.npy)."""
    path = os.path.join(MODEL_DIR, 'reward_history.npy')
    if not os.path.exists(path):
        return {"rewards": []}
    arr = np.load(path)
    # Downsample so the payload stays small for long training runs
    if len(arr) > 300:
        idx = np.linspace(0, len(arr) - 1, 300).astype(int)
        arr = arr[idx]
    return {"rewards": [round(float(x), 1) for x in arr]}


class TrainRequest(BaseModel):
    episodes: int = 100


@app.post("/api/train")
def train(req: TrainRequest):
    """Runs additional training episodes synchronously (kept small; this is a demo)."""
    episodes = max(1, min(req.episodes, 300))
    start_ep = sim.coordinator.reward_history and len(sim.coordinator.reward_history) or 0
    for a in sim.ward_agents.values():
        a.epsilon = max(a.epsilon, 0.3)  # allow some exploration while retraining
    for ep in range(start_ep, start_ep + episodes):
        sim.coordinator.run_episode(ep, train=True, verbose=False)
    sim.coordinator.save_all()
    np.save(os.path.join(MODEL_DIR, 'reward_history.npy'),
            np.array(sim.coordinator.reward_history))
    for a in sim.ward_agents.values():
        a.epsilon = 0.0
    return {"trained_episodes": episodes,
            "final_avg_reward": round(float(np.mean(sim.coordinator.reward_history[-20:])), 1)}


@app.websocket("/ws/run")
async def ws_run(websocket: WebSocket):
    """
    Streams one full episode (24 timesteps) step-by-step over the socket,
    so the frontend can animate patients arriving in real time.
    Client may send {"delay_ms": 400} to control playback speed.
    """
    await websocket.accept()
    delay_ms = 500
    try:
        init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=2.0)
        delay_ms = int(init_msg.get('delay_ms', 500))
    except Exception:
        pass

    snap = sim.reset_episode()
    await websocket.send_json({"type": "reset", "data": snap})

    try:
        while not sim.done:
            entry = sim.take_step(train=False)
            await websocket.send_json({"type": "step", "data": entry})
            await asyncio.sleep(delay_ms / 1000.0)
        await websocket.send_json({"type": "done", "data": sim.snapshot()})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass


# Serve the frontend as static files (if present) at the root path.
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
