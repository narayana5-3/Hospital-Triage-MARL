"""
Headless training script. Run this once before starting the API server
so the demo uses trained (low-epsilon) agents instead of random ones.

Usage:
    python train.py --episodes 500
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.hospital_env import MultiWardHospitalEnv
from env.patient_generator import SyntheticPatientGenerator
from agents.ward_agent import WardAgent
from coordinator.message_bus import MessageBus
from coordinator.hospital_coordinator import HospitalCoordinator
from planner.llm_planner import LLMCoordinatorPlanner

logging.basicConfig(level=logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verbose-every', type=int, default=50)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model_dir = os.path.join(BASE_DIR, 'models')
    log_dir = os.path.join(BASE_DIR, 'logs')
    memory_dir = os.path.join(BASE_DIR, 'memory')

    env = MultiWardHospitalEnv()
    patient_gen = SyntheticPatientGenerator(seed=42)

    ward_agents = {
        name: WardAgent(ward_name=name, device=device,
                         model_dir=model_dir, memory_dir=memory_dir)
        for name in ['Emergency', 'General', 'ICU']
    }

    planner = LLMCoordinatorPlanner(plan_interval=10)
    bus = MessageBus(log_dir=log_dir)

    coordinator = HospitalCoordinator(
        env=env, patient_gen=patient_gen, ward_agents=ward_agents,
        planner=planner, message_bus=bus, log_dir=log_dir, model_dir=model_dir,
    )

    if args.resume:
        coordinator.load_all()

    start_ep = len(coordinator.reward_history)
    print(f'LLM backend : {planner.stats["backend"]}')
    print(f'Training episodes {start_ep} -> {args.episodes}')

    for ep in range(start_ep, args.episodes):
        total_reward = coordinator.run_episode(ep, train=True, verbose=False)
        if ep % args.verbose_every == 0 or ep == args.episodes - 1:
            avgs = {n: round(np.mean(a.reward_history[-20:]), 1)
                    if len(a.reward_history) else 0.0
                    for n, a in ward_agents.items()}
            print(f'[Ep {ep:4d}] total_reward={total_reward:7.1f}  '
                  f'avg20={avgs}  eps(EM)={ward_agents["Emergency"].epsilon:.3f}')

    coordinator.save_all()
    np.save(os.path.join(model_dir, 'reward_history.npy'),
            np.array(coordinator.reward_history))
    print('\nTraining complete.')
    print(f'Final avg total reward (last 50): '
          f'{np.mean(coordinator.reward_history[-50:]):.1f}')


if __name__ == '__main__':
    main()
