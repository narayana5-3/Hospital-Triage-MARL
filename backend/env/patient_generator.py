import numpy as np

# NEWS2 severity scoring

COMPLAINT_WEIGHTS = {
    'cardiac_arrest':       3,
    'major_trauma':         3,
    'stroke':               3,
    'chest_pain':           2,
    'difficulty_breathing': 2,
    'seizure':              2,
    'abdominal_pain':       1,
    'infection':            1,
    'minor_injury':         0,
    'general_complaint':    0,
}

COMPLAINTS_BY_BAND = {
    'low':      ['minor_injury', 'general_complaint'],
    'moderate': ['abdominal_pain', 'infection', 'chest_pain'],
    'critical': ['cardiac_arrest', 'major_trauma', 'stroke', 'difficulty_breathing', 'seizure'],
}

AVPU_SCORES = {
    'alert':        0,
    'voice':        1,
    'pain':         2,
    'unresponsive': 3,
}


def score_heart_rate(hr):
    if hr < 40 or hr > 130: return 3
    if hr <= 50 or hr >= 111: return 2
    if hr <= 60 or hr >= 101: return 1
    return 0


def score_bp(sbp):
    if sbp < 90:   return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 159: return 0
    if sbp <= 179: return 1
    return 3


def score_spo2(spo2):
    if spo2 < 88:  return 3
    if spo2 <= 91: return 2
    if spo2 <= 93: return 1
    return 0


def score_resp_rate(rr):
    if rr < 8 or rr >= 30: return 3
    if rr <= 11 or rr >= 25: return 2
    if rr >= 21: return 1
    return 0


def score_temperature(temp):
    if temp < 35.0 or temp > 39.5: return 2
    if temp < 36.0 or temp > 38.5: return 1
    return 0


def score_consciousness(avpu):
    return AVPU_SCORES.get(avpu, 0)


def score_age(age):
    if age >= 80: return 2
    if age >= 65: return 1
    return 0


def calculate_severity(vitals):
    """
    Convert a vitals dict into a severity score 0.0-1.0.
    Based on NEWS2 clinical scoring (Royal College of Physicians UK).
    Max raw score ~22. Normalised by dividing by 14.
    """
    raw = 0
    raw += score_heart_rate(vitals['heart_rate'])
    raw += score_bp(vitals['systolic_bp'])
    raw += score_spo2(vitals['spo2'])
    raw += score_resp_rate(vitals['resp_rate'])
    raw += score_temperature(vitals['temperature'])
    raw += score_consciousness(vitals['consciousness'])
    raw += score_age(vitals['age'])
    raw += COMPLAINT_WEIGHTS.get(vitals['chief_complaint'], 0)

    severity = float(np.clip(raw / 14.0, 0.05, 0.98))
    return round(severity, 3)


def severity_band(severity):
    if severity <= 0.35: return 'low'
    if severity <= 0.65: return 'moderate'
    return 'critical'


class SyntheticPatientGenerator:
    """
    Generates realistic synthetic patients with vital signs.
    Severity distribution is controlled to produce ~20% critical,
    ~35% moderate, ~45% low - matching realistic emergency dept arrivals.
    """

    BAND_PROBS = [0.45, 0.35, 0.20]   # low / moderate / critical

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._count = 0

    def generate(self, force_band=None):
        band = force_band or self.rng.choice(
            ['low', 'moderate', 'critical'], p=self.BAND_PROBS)
        vitals = self._sample_vitals(band)
        severity = calculate_severity(vitals)
        # Recalculate band from actual computed severity
        actual_band = severity_band(severity)
        self._count += 1
        return {
            'id':              self._count,
            'arrival_time':    self._count,
            'vitals':          vitals,
            'severity':        severity,
            'severity_band':   actual_band,
            'critical':        actual_band == 'critical',
            'source_band':     band,
        }

    def _sample_vitals(self, band):
        r = self.rng
        if band == 'low':
            hr    = int(r.integers(65, 96))
            sbp   = int(r.integers(110, 141))
            spo2  = int(r.integers(96, 100))
            rr    = int(r.integers(14, 19))
            temp  = round(float(r.uniform(36.5, 37.5)), 1)
            avpu  = 'alert'
            age   = int(r.integers(18, 65))
            comp  = str(r.choice(COMPLAINTS_BY_BAND['low']))
        elif band == 'moderate':
            hr    = int(r.choice([
                r.integers(50, 66), r.integers(100, 116)]))
            sbp   = int(r.choice([
                r.integers(95, 111), r.integers(150, 166)]))
            spo2  = int(r.integers(92, 96))
            rr    = int(r.integers(20, 26))
            temp  = round(float(r.choice([
                r.uniform(35.5, 36.4), r.uniform(38.0, 39.1)])), 1)
            avpu  = str(r.choice(['alert', 'voice']))
            age   = int(r.integers(40, 80))
            comp  = str(r.choice(COMPLAINTS_BY_BAND['moderate']))
        else:  # critical
            hr    = int(r.choice([
                r.integers(30, 46), r.integers(120, 141)]))
            sbp   = int(r.choice([
                r.integers(60, 91), r.integers(175, 201)]))
            spo2  = int(r.integers(78, 91))
            rr    = int(r.choice([
                r.integers(5, 9), r.integers(28, 36)]))
            temp  = round(float(r.choice([
                r.uniform(33.0, 35.1), r.uniform(39.5, 41.0)])), 1)
            avpu  = str(r.choice(['voice', 'pain', 'unresponsive']))
            age   = int(r.integers(50, 90))
            comp  = str(r.choice(COMPLAINTS_BY_BAND['critical']))

        return {
            'heart_rate':       hr,
            'systolic_bp':      sbp,
            'spo2':             spo2,
            'resp_rate':        rr,
            'temperature':      temp,
            'consciousness':    avpu,
            'age':              age,
            'chief_complaint':  comp,
        }
