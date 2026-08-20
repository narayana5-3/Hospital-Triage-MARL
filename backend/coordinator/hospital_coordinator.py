import os, logging, json
import numpy as np
from datetime import datetime

logger = logging.getLogger('Coordinator')


class AuditLogger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.path = os.path.join(log_dir, f'coordinator_{ts}.jsonl')

    def log(self, episode, step, patient, plan, em_action,
            em_action_name, reward, ward_states, messages):
        entry = {
            'episode': episode, 'step': step,
            'ts': datetime.now().isoformat(),
            'patient': {k: v for k, v in patient.items() if k != 'vitals'},
            'llm_plan': plan,
            'em_action': em_action,
            'em_action_name': em_action_name,
            'reward': float(reward),
            'ward_states': ward_states,
            'messages': messages,
        }
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + '\n')


class HospitalCoordinator:
    """
    Each timestep:
      1. Patient generated and injected into environment.
      2. Ward messages posted to bus (capacity, escalation signals).
      3. LLM planner called for triage recommendation.
      4. Coordinator hint encoded and passed to all agents.
      5. Emergency agent decides: Keep(0) / TransferGeneral(1) / TransferICU(2).
      6. General and ICU agents handle pending transfers from their wards.
      7. Environment executes triage and updates all patient severities.
      8. Rewards distributed to all agents.
      9. Audit log written.
    """

    def __init__(self, env, patient_gen, ward_agents, planner,
                 message_bus, log_dir='logs', model_dir='models'):
        self.env         = env
        self.patient_gen = patient_gen
        self.agents      = ward_agents
        self.planner     = planner
        self.bus         = message_bus
        self.auditor     = AuditLogger(log_dir)
        self.model_dir   = model_dir
        self.reward_history = []

    # Main step

    def step(self, episode, step_idx, train=True, verbose=False):
        # 1. Generate patient and inject into env
        patient = self.patient_gen.generate()
        self.env.set_patient(patient)

        global_state = self.env.global_state()
        local_states = self.env.local_states()

        # 2. Post ward messages
        self._post_ward_messages()
        pending_messages = self.bus.pending_dicts()

        # 3. LLM plan
        context = self.env.get_context()
        plan    = self.planner.plan(context, global_state, patient, step_idx)
        hint    = self.planner.encode_hint(plan)
        coord_hint = hint

        # 4. Emergency agent decides triage action
        em_agent = self.agents['Emergency']
        em_action, em_aug, em_conf = em_agent.decide(
            local_states['Emergency'], coord_hint)

        # Guide early training with LLM recommendation while still exploring
        if em_agent.epsilon > 0.5:
            llm_action = plan.get('recommended_action', 0)
            if np.random.random() < 0.3:
                em_action = int(llm_action)

        # 5. Execute triage in environment
        triage_reward, done, info = self.env.triage(em_action)

        # 6. Handle secondary transfers from General/ICU agents
        secondary_reward = self._handle_secondary_transfers(
            local_states, coord_hint, train)

        total_reward = triage_reward + secondary_reward

        # 7. Distribute rewards and train
        new_local = self.env.local_states()
        new_hint  = self.planner.encode_hint(plan)

        if train:
            em_agent.learn(triage_reward, new_local['Emergency'],
                           new_hint, done)
            self.agents['General'].learn(
                secondary_reward * 0.5, new_local['General'], new_hint, done)
            self.agents['ICU'].learn(
                secondary_reward * 0.5, new_local['ICU'], new_hint, done)

        # 8. Audit log
        ward_states_summary = {
            n: {'free': w.free, 'occupied': w.occupied,
                'avg_sev': round(w.avg_severity, 3)}
            for n, w in self.env.wards.items()}
        em_action_name = em_agent.action_names.get(em_action, str(em_action))

        self.auditor.log(episode, step_idx, patient, plan,
                         em_action, em_action_name, total_reward,
                         ward_states_summary, self.bus.summarise())
        self.bus.flush()

        if verbose:
            band = patient.get('severity_band', '?').upper()
            print(f"  t{step_idx:02d} | sev={patient['severity']:.3f} "
                  f"[{band}] | LLM->{plan.get('recommended_action','?')} "
                  f"| EM->{em_action_name:<20} "
                  f"| reward={total_reward:+.0f} "
                  f"| [{plan.get('source','?')}]")

        return total_reward, done, {
            'em_action': em_action,
            'em_action_name': em_action_name,
            'em_confidence': round(em_conf, 3),
            'patient': patient,
            'plan': plan,
            'outcome': info.get('outcome', ''),
            'messages': pending_messages,
            'ward_states': self.env.wards_summary(),
            'time': self.env.time,
        }

    # Secondary transfer handling

    def _handle_secondary_transfers(self, local_states, coord_hint, train):
        """
        General and ICU agents inspect their own ward for patients
        who need escalation or de-escalation and request transfers.
        """
        total = 0.0

        gen_agent  = self.agents['General']
        gen_ward   = self.env.wards['General']
        gen_action, _, _ = gen_agent.decide(
            local_states['General'], coord_hint)

        if gen_action == 2:  # Escalate to ICU
            patients_needing_icu = gen_ward.patients_need_escalate
            for p in patients_needing_icu[:1]:
                r = self.env.execute_transfer(p.id, 'General', 'ICU')
                total += r
                if r > 0:
                    self.bus.post('General', 'ESCALATION_NEEDED',
                                  {'patient_id': p.id,
                                   'severity': p.severity})

        icu_agent = self.agents['ICU']
        icu_ward  = self.env.wards['ICU']
        icu_action, _, _ = icu_agent.decide(
            local_states['ICU'], coord_hint)

        if icu_action == 2:  # De-escalate to General
            patients_ready = icu_ward.patients_ready_deescalate
            for p in patients_ready[:1]:
                r = self.env.execute_transfer(p.id, 'ICU', 'General')
                total += r
                if r > 0:
                    self.bus.post('ICU', 'DEESCALATION_READY',
                                  {'patient_id': p.id,
                                   'severity': p.severity})

        return total

    # Ward message posting

    def _post_ward_messages(self):
        for name, ward in self.env.wards.items():
            if ward.occupancy_rate >= 0.8:
                self.bus.post(name, 'CAPACITY_ALERT',
                              {'occupancy': round(ward.occupancy_rate, 2)})
            elif ward.occupancy_rate <= 0.4 and ward.free >= 2:
                self.bus.post(name, 'RESOURCE_OFFER',
                              {'free_beds': ward.free})

        for p in self.env.wards['General'].patients_need_escalate:
            self.bus.post('General', 'ESCALATION_NEEDED',
                          {'patient_id': p.id, 'severity': round(p.severity, 3)})

        for p in self.env.wards['ICU'].patients_ready_deescalate:
            self.bus.post('ICU', 'DEESCALATION_READY',
                          {'patient_id': p.id, 'severity': round(p.severity, 3)})

    # Episode loop

    def run_episode(self, episode, train=True, verbose=False):
        self.env.reset()
        total_reward = 0.0
        for step_idx in range(24):
            reward, done, info = self.step(
                episode, step_idx, train, verbose)
            total_reward += reward
            if done:
                break
        for name, agent in self.agents.items():
            agent.end_episode(episode, self.env.wards[name])
        self.reward_history.append(total_reward)
        return total_reward

    # Persistence

    def save_all(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for agent in self.agents.values():
            agent.save()
        np.save(os.path.join(self.model_dir, 'coordinator_rewards.npy'),
                np.array(self.reward_history))
        print(f'All agents saved to {self.model_dir}/')

    def load_all(self):
        for agent in self.agents.values():
            agent.load()
        rpath = os.path.join(self.model_dir, 'coordinator_rewards.npy')
        if os.path.exists(rpath):
            self.reward_history = list(np.load(rpath))
        print('All agents loaded.')
