import os, random, logging, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

LOCAL_OBS_DIM  = 6
COORD_HINT_DIM = 4
AUGMENTED_DIM  = LOCAL_OBS_DIM + COORD_HINT_DIM   # 10

# Actions differ per ward type
EM_ACTIONS  = {0: 'Keep_Emergency', 1: 'Transfer_General', 2: 'Transfer_ICU'}
GEN_ACTIONS = {0: 'Accept',         1: 'Decline',          2: 'Escalate_ICU'}
ICU_ACTIONS = {0: 'Accept',         1: 'Decline',          2: 'Deescalate_General'}
ACTION_DIM  = 3


class DQN(nn.Module):
    def __init__(self, state_dim=AUGMENTED_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, action_dim),
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, maxlen=6000):
        self._buf = deque(maxlen=maxlen)

    def push(self, s, a, r, ns, done):
        self._buf.append((s, a, r, ns, float(done)))

    def sample(self, n):
        batch = random.sample(self._buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.int64),
                np.array(r,  dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d,  dtype=np.float32))

    def __len__(self): return len(self._buf)


class WardEpisodicMemory:
    def __init__(self, ward_name, save_dir='memory'):
        os.makedirs(save_dir, exist_ok=True)
        self.path  = os.path.join(save_dir, f'{ward_name.lower()}_episodic.jsonl')
        self._hist = deque(maxlen=500)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    try: self._hist.append(json.loads(line))
                    except: pass

    def record(self, episode, total_reward, admitted, rejected, transfers):
        entry = {'episode': episode, 'ts': datetime.now().isoformat(),
                 'total_reward': round(float(total_reward), 2),
                 'admitted': admitted, 'rejected': rejected,
                 'transfers': transfers}
        self._hist.append(entry)
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return entry

    def recent_avg(self, n=20):
        recent = list(self._hist)[-n:]
        return round(np.mean([e['total_reward'] for e in recent]), 2) if recent else 0.0


class WardAgent:
    """
    Autonomous DQN agent for a single hospital ward.

    Emergency agent actions : 0=Keep, 1=TransferGeneral, 2=TransferICU
    General agent actions   : 0=Accept, 1=Decline, 2=EscalateICU
    ICU agent actions       : 0=Accept, 1=Decline, 2=DeescalateGeneral
    """

    ACTION_MAPS = {
        'Emergency': EM_ACTIONS,
        'General':   GEN_ACTIONS,
        'ICU':       ICU_ACTIONS,
    }

    def __init__(self, ward_name, device,
                 gamma=0.99, lr=0.001,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995,
                 batch_size=64, target_update=10,
                 model_dir='models', memory_dir='memory'):

        self.name          = ward_name
        self.device        = device
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.model_dir     = model_dir
        self._update_count = 0
        self.action_names  = self.ACTION_MAPS[ward_name]

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn    = nn.MSELoss()

        self.replay         = ReplayBuffer()
        self.memory         = WardEpisodicMemory(ward_name, memory_dir)
        self.reward_history = []
        self._ep_reward     = 0.0
        self._last_state    = None
        self._last_action   = None

    def augment(self, local_obs, coord_hint):
        return np.concatenate([local_obs, coord_hint], dtype=np.float32)

    def select_action(self, aug_state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        with torch.no_grad():
            t = torch.FloatTensor(aug_state).unsqueeze(0).to(self.device)
            return int(torch.argmax(self.policy_net(t)).item())

    def decide(self, local_obs, coord_hint):
        """
        Returns (action, aug_state, confidence).
        Called every timestep for all three agents.
        """
        aug    = self.augment(local_obs, coord_hint)
        action = self.select_action(aug)
        with torch.no_grad():
            t     = torch.FloatTensor(aug).unsqueeze(0).to(self.device)
            qvals = self.policy_net(t).cpu().numpy()[0]
        exp_q      = np.exp(qvals - np.max(qvals))
        confidence = float(exp_q[action] / exp_q.sum())
        self._last_state  = aug
        self._last_action = action
        return action, aug, confidence

    def learn(self, reward, next_obs, coord_hint, done):
        if self._last_state is None:
            return
        next_aug = self.augment(next_obs, coord_hint)
        self.replay.push(self._last_state, self._last_action,
                         reward, next_aug, done)
        self._ep_reward += reward
        self._train_step()

    def _train_step(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, ns, d = self.replay.sample(self.batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)
        q      = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target_net(ns).max(1)[0]
        target = r + self.gamma * next_q * (1 - d)
        loss   = self.loss_fn(q, target.detach())
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self._update_count += 1
        if self._update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self, episode, ward_state):
        ep_r = self._ep_reward
        self.reward_history.append(ep_r)
        self.memory.record(
            episode, ep_r,
            ward_state.admitted_this_ep,
            ward_state.rejected_this_ep,
            ward_state.transfers_out)
        self.epsilon    = max(self.epsilon_min,
                              self.epsilon * self.epsilon_decay)
        self._ep_reward = 0.0
        self._last_state  = None
        self._last_action = None
        return ep_r

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f'{self.name.lower()}_agent.pt')
        torch.save({'policy': self.policy_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'optim':  self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'reward_history': self.reward_history}, path)

    def load(self):
        path = os.path.join(self.model_dir, f'{self.name.lower()}_agent.pt')
        if not os.path.exists(path):
            print(f'No saved model for {self.name}.'); return
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck['policy'])
        self.target_net.load_state_dict(ck['target'])
        self.optimizer.load_state_dict(ck['optim'])
        self.epsilon        = ck.get('epsilon', self.epsilon_min)
        self.reward_history = ck.get('reward_history', [])
        print(f'{self.name} agent loaded.')
