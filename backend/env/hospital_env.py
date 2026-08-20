import numpy as np

#  Reward table
REWARD = {
    'correct_triage_low_to_general':   15,
    'correct_triage_moderate_em':      10,
    'correct_triage_critical_to_icu':  25,
    'wrong_triage_critical_kept_em':  -20,
    'wrong_triage_stable_kept_em':     -8,
    'transfer_accepted':               10,
    'transfer_refused':               -10,
    'deterioration_wrong_ward':       -15,
    'icu_deescalate_to_general':        8,
    'critical_missed':                -30,
    'transfer_delay_per_step':         -2,
}

WARD_CAPS    = {'ICU': 5, 'General': 15, 'Emergency': 10}
WARD_BANDS   = {'ICU': 'critical', 'General': 'low', 'Emergency': 'moderate'}
MATCH_SCORE  = {
    ('ICU',       'critical'): +1,
    ('ICU',       'moderate'):  0,
    ('ICU',       'low'):      -1,
    ('Emergency', 'moderate'): +1,
    ('Emergency', 'critical'):  0,
    ('Emergency', 'low'):       0,
    ('General',   'low'):      +1,
    ('General',   'moderate'):  0,
    ('General',   'critical'): -1,
}

def severity_band(s):
    if s <= 0.35: return 'low'
    if s <= 0.65: return 'moderate'
    return 'critical'


class AdmittedPatient:
    """Tracks a patient currently occupying a bed."""
    def __init__(self, patient_dict, ward):
        self.id              = patient_dict['id']
        self.severity        = patient_dict['severity']
        self.original_sev    = patient_dict['severity']
        self.ward            = ward
        self.steps_in_ward   = 0
        self.transfer_pending = False
        self.transfer_target  = None
        self.steps_waiting_transfer = 0
        self.vitals          = patient_dict.get('vitals', {})

    def band(self):
        return severity_band(self.severity)

    def update_severity(self, ward_occupancy, rng):
        """
        Update severity based on ward-patient match quality,
        capacity stress, and random clinical factor.
        """
        match  = MATCH_SCORE.get((self.ward, self.band()), 0)

        # Capacity stress
        if   ward_occupancy < 0.50: stress = 0.00
        elif ward_occupancy < 0.70: stress = 0.10
        elif ward_occupancy < 0.90: stress = 0.25
        else:                       stress = 0.40

        # Transfer delay penalty
        delay_penalty = 0.0
        if self.transfer_pending and self.steps_waiting_transfer > 2:
            delay_penalty = 0.02 * self.steps_waiting_transfer

        # Base change: positive = improvement (severity decreases)
        base_change = (match * 0.04) - stress - delay_penalty

        # Random clinical factor
        random_factor = float(rng.normal(0, 0.03))

        # Apply: severity DECREASES when improving (base_change positive)
        delta = -base_change + random_factor
        self.severity = float(np.clip(self.severity + delta, 0.05, 0.98))
        self.severity = round(self.severity, 3)
        self.steps_in_ward += 1

        if self.transfer_pending:
            self.steps_waiting_transfer += 1


class WardState:
    def __init__(self, name, capacity):
        self.name       = name
        self.capacity   = capacity
        self.patients   = []           # list of AdmittedPatient
        self.admitted_this_ep  = 0
        self.rejected_this_ep  = 0
        self.transfers_in      = 0
        self.transfers_out     = 0
        self.deteriorations    = 0

    def reset(self):
        self.patients          = []
        self.admitted_this_ep  = 0
        self.rejected_this_ep  = 0
        self.transfers_in      = 0
        self.transfers_out     = 0
        self.deteriorations    = 0

    @property
    def occupied(self): return len(self.patients)

    @property
    def free(self): return self.capacity - self.occupied

    @property
    def occupancy_rate(self):
        return self.occupied / self.capacity if self.capacity else 0.0

    @property
    def avg_severity(self):
        if not self.patients: return 0.0
        return round(float(np.mean([p.severity for p in self.patients])), 3)

    @property
    def patients_ready_deescalate(self):
        return [p for p in self.patients if p.ward == 'ICU' and p.severity < 0.35]

    @property
    def patients_need_escalate(self):
        return [p for p in self.patients if p.ward == 'General' and p.severity > 0.65]

    def admit(self, patient_dict):
        ap = AdmittedPatient(patient_dict, self.name)
        self.patients.append(ap)
        self.admitted_this_ep += 1
        return ap

    def remove_patient(self, patient_id):
        before = len(self.patients)
        self.patients = [p for p in self.patients if p.id != patient_id]
        return len(self.patients) < before

    def observe(self, time_frac, incoming_severity=0.5, bus_hint=None):
        """6-float local observation for the ward agent."""
        bus_hint = bus_hint or [0.0, 0.0]
        return np.array([
            self.free / self.capacity,
            self.occupancy_rate,
            incoming_severity,
            self.avg_severity,
            time_frac,
            bus_hint[0],   # pressure from other wards
        ], dtype=np.float32)

    def to_dict(self):
        return {
            'name': self.name,
            'capacity': self.capacity,
            'occupied': self.occupied,
            'free': self.free,
            'occupancy_rate': round(self.occupancy_rate, 3),
            'avg_severity': self.avg_severity,
            'admitted_this_ep': self.admitted_this_ep,
            'rejected_this_ep': self.rejected_this_ep,
            'transfers_in': self.transfers_in,
            'transfers_out': self.transfers_out,
            'deteriorations': self.deteriorations,
            'patients': [
                {'id': p.id, 'severity': p.severity, 'band': p.band(),
                 'steps_in_ward': p.steps_in_ward,
                 'transfer_pending': p.transfer_pending,
                 'transfer_target': p.transfer_target}
                for p in self.patients
            ],
        }


class MultiWardHospitalEnv:
    """
    Triage-based hospital environment.
    All patients enter through Emergency first.
    Emergency agent triages: Keep / TransferGeneral / TransferICU
    Severity updates every timestep based on match quality + capacity stress.
    """

    def __init__(self):
        self.wards = {
            n: WardState(n, c) for n, c in WARD_CAPS.items()}
        self.rng                   = np.random.default_rng()
        self.time                  = 0
        self.current_patient       = None
        self.episode_log           = []
        self.total_critical_missed = 0
        self.total_admitted        = 0
        self._patient_id_counter   = 0

    def seed(self, s):
        self.rng = np.random.default_rng(s)

    def reset(self):
        for w in self.wards.values():
            w.reset()
        self.time                  = 0
        self.current_patient       = None
        self.episode_log           = []
        self.total_critical_missed = 0
        self.total_admitted        = 0
        self._patient_id_counter   = 0
        return self.global_state(), self.local_states()

    # Patient management

    def set_patient(self, patient_dict):
        """Called by coordinator to set next patient for triage."""
        self.current_patient = patient_dict

    # Triage action (Emergency agent decides)

    def triage(self, action):
        """
        Emergency agent triages current patient.
        action: 0=Keep in Emergency, 1=Transfer to General, 2=Transfer to ICU
        Returns (reward, done, info)
        """
        p    = self.current_patient
        sev  = p['severity']
        band = severity_band(sev)
        info = {'action': action, 'patient_id': p['id'],
                'severity': sev, 'band': band, 'outcome': ''}

        if action == 0:   # Keep in Emergency
            reward, info = self._admit_to('Emergency', p, band, info)
        elif action == 1: # Transfer to General
            reward, info = self._admit_to('General', p, band, info)
        elif action == 2: # Transfer to ICU
            reward, info = self._admit_to('ICU', p, band, info)
        else:
            reward = -5; info['outcome'] = 'invalid_action'

        # Step time and update all patient severities
        self.time += 1
        secondary_rewards = self._update_all_patients()
        reward += secondary_rewards

        done = self.time >= 24
        self.episode_log.append({**info, 'time': self.time,
                                  'reward': reward})
        return reward, done, info

    def _admit_to(self, ward_name, patient, band, info):
        ward = self.wards[ward_name]
        if ward.free > 0:
            ward.admit(patient)
            self.total_admitted += 1
            reward = self._triage_reward(ward_name, band)
            info['outcome'] = f'admitted_{ward_name.lower()}'
        else:
            ward.rejected_this_ep += 1
            if band == 'critical':
                reward = REWARD['critical_missed']
                self.total_critical_missed += 1
                info['outcome'] = 'critical_missed'
            else:
                reward = REWARD['transfer_refused']
                info['outcome'] = 'transfer_refused'
        return reward, info

    def _triage_reward(self, ward_name, band):
        if ward_name == 'Emergency' and band == 'moderate':
            return REWARD['correct_triage_moderate_em']
        if ward_name == 'General'   and band == 'low':
            return REWARD['correct_triage_low_to_general']
        if ward_name == 'ICU'       and band == 'critical':
            return REWARD['correct_triage_critical_to_icu']
        if ward_name == 'Emergency' and band == 'critical':
            return REWARD['wrong_triage_critical_kept_em']
        if ward_name == 'Emergency' and band == 'low':
            return REWARD['wrong_triage_stable_kept_em']
        # Misrouted but not worst case
        return -5

    # Secondary transfers (General/ICU agents)

    def execute_transfer(self, patient_id, from_ward, to_ward):
        """
        General or ICU agent requests a transfer for an existing patient.
        Returns reward delta.
        """
        src = self.wards[from_ward]
        dst = self.wards[to_ward]
        patient = next((p for p in src.patients if p.id == patient_id), None)
        if patient is None:
            return -5

        if dst.free > 0:
            src.remove_patient(patient_id)
            src.transfers_out += 1
            patient.ward = to_ward
            patient.transfer_pending = False
            patient.steps_waiting_transfer = 0
            dst.patients.append(patient)
            dst.transfers_in += 1
            if from_ward == 'ICU' and to_ward == 'General':
                return REWARD['icu_deescalate_to_general']
            return REWARD['transfer_accepted']
        else:
            # Mark as pending - will accumulate delay penalty
            patient.transfer_pending = True
            patient.transfer_target  = to_ward
            return REWARD['transfer_refused']

    # Per-timestep severity updates

    def _update_all_patients(self):
        """Update severity for every admitted patient. Returns total reward delta."""
        total_reward = 0.0
        for ward in self.wards.values():
            for patient in ward.patients:
                old_band = patient.band()
                patient.update_severity(ward.occupancy_rate, self.rng)
                new_band = patient.band()

                # Auto-flag patients who crossed band boundaries
                if old_band == 'critical' and new_band in ('low', 'moderate'):
                    if ward.name == 'ICU':
                        patient.transfer_pending = True
                        patient.transfer_target  = 'General'
                elif old_band in ('low', 'moderate') and new_band == 'critical':
                    if ward.name == 'General':
                        patient.transfer_pending = True
                        patient.transfer_target  = 'ICU'
                        ward.deteriorations += 1
                        total_reward += REWARD['deterioration_wrong_ward']

        # Random discharges (simulates treatment completion)
        for name, ward in self.wards.items():
            n_discharge = int(self.rng.integers(0, 3 if name == 'General' else 2))
            n_discharge = min(n_discharge, len(ward.patients))
            for _ in range(n_discharge):
                if ward.patients:
                    ward.patients.pop(0)

        return total_reward

    # Observation builders

    def global_state(self):
        """14-float global state for coordinator / LLM context."""
        w = self.wards
        p = self.current_patient or {'severity': 0.5, 'critical': False}
        return np.array([
            w['ICU'].free        / WARD_CAPS['ICU'],
            w['ICU'].occupancy_rate,
            w['ICU'].avg_severity,
            w['General'].free    / WARD_CAPS['General'],
            w['General'].occupancy_rate,
            w['General'].avg_severity,
            w['Emergency'].free  / WARD_CAPS['Emergency'],
            w['Emergency'].occupancy_rate,
            w['Emergency'].avg_severity,
            float(p['severity']),
            float(p.get('critical', False)),
            self.time / 24.0,
            len([pt for w2 in self.wards.values()
                 for pt in w2.patients if pt.transfer_pending]) / 10.0,
            float(self.total_critical_missed) / 10.0,
        ], dtype=np.float32)

    def local_states(self):
        """Per-ward 6-float observations."""
        tf  = self.time / 24.0
        p   = self.current_patient or {'severity': 0.5}
        sev = p['severity']
        return {
            name: w.observe(tf, sev) for name, w in self.wards.items()}

    def get_context(self):
        """Natural language context for LLM planner."""
        lines = [f'Hospital triage status at hour {self.time}/24:']
        for name, w in self.wards.items():
            pending = sum(1 for pt in w.patients if pt.transfer_pending)
            lines.append(
                f'  {name:<10}: {w.free}/{w.capacity} free '
                f'({round(w.occupancy_rate*100)}% occupied) '
                f'avg_sev={w.avg_severity:.2f} pending_transfers={pending}')
        p = self.current_patient
        if p:
            lbl = p.get('severity_band', 'unknown').upper()
            lines.append(
                f'  Incoming patient: severity={p["severity"]:.3f} [{lbl}]')
            v = p.get('vitals', {})
            if v:
                lines.append(
                    f'  Vitals: HR={v.get("heart_rate","?")} '
                    f'BP={v.get("systolic_bp","?")} '
                    f'SpO2={v.get("spo2","?")}% '
                    f'RR={v.get("resp_rate","?")} '
                    f'Temp={v.get("temperature","?")} '
                    f'AVPU={v.get("consciousness","?")}')
        lines.append(f'  Critical missed: {self.total_critical_missed}')
        return '\n'.join(lines)

    def wards_summary(self):
        return {name: w.to_dict() for name, w in self.wards.items()}
