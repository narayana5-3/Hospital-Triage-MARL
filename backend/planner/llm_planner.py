import json, os, time, urllib.request, urllib.error, logging

logger = logging.getLogger(__name__)

GROQ_URL      = 'https://api.groq.com/openai/v1/chat/completions'
DEFAULT_MODEL = 'llama-3.1-8b-instant'
_RATE_LIMIT   = 2.1
_last_call    = 0.0

SYSTEM_PROMPT = """You are the AI triage coordinator of a hospital.
All patients enter through Emergency. Your job is to recommend
the correct routing based on patient severity and ward capacity.
Respond ONLY with valid JSON. No markdown, no extra text.

Schema:
{"recommended_action": 0|1|2,
 "risk_level": "low"|"moderate"|"high"|"critical",
 "allow_transfer": true|false,
 "reasoning": "one sentence",
 "confidence": 0.0}

Action mapping for Emergency agent:
  0 = Keep patient in Emergency (moderate severity 0.36-0.65)
  1 = Transfer to General Ward  (low severity 0.00-0.35)
  2 = Transfer to ICU           (critical severity 0.66-1.00)

Rules:
- Severity > 0.65: recommend action 2 (ICU) unless ICU is full.
- Severity 0.36-0.65: recommend action 0 (keep Emergency).
- Severity < 0.36: recommend action 1 (General) unless General is full.
- If target ward is full, recommend next best available option.
- Consider pending transfers and deteriorating patients in your reasoning."""


def _load_key():
    return os.environ.get('GROQ_API_KEY', '').strip() or None


def _fallback(global_state, patient):
    sev  = patient.get('severity', 0.5)
    band = patient.get('severity_band', 'moderate')
    icu_free = global_state[0] * 5
    gen_free = global_state[3] * 15
    if band == 'critical':
        action = 2 if icu_free > 0 else 0
        risk   = 'critical'
    elif band == 'low':
        action = 1 if gen_free > 0 else 0
        risk   = 'low'
    else:
        action = 0
        risk   = 'moderate'
    return {'recommended_action': action, 'risk_level': risk,
            'allow_transfer': False,
            'reasoning': 'Rule-based fallback.',
            'confidence': 0.75, 'source': 'fallback'}


def _call_groq(context, api_key, model):
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _RATE_LIMIT:
        time.sleep(_RATE_LIMIT - elapsed)
    payload = json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',
             'content': context + '\nRespond ONLY with the JSON object.'}],
        'temperature': 0.2, 'max_tokens': 200,
        'response_format': {'type': 'json_object'},
    }).encode('utf-8')
    req = urllib.request.Request(
        GROQ_URL, data=payload,
        headers={'Content-Type': 'application/json',
                 'Authorization': f'Bearer {api_key}'})
    _last_call = time.time()
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read().decode('utf-8'))
    raw = result['choices'][0]['message']['content'].strip()
    raw = raw.lstrip('```json').lstrip('```').rstrip('```').strip()
    return json.loads(raw)


class LLMCoordinatorPlanner:
    def __init__(self, model=DEFAULT_MODEL, plan_interval=4):
        self.model         = model
        self.plan_interval = plan_interval
        self.api_key       = _load_key()
        self.last_plan     = None
        self.call_count    = 0
        self.fail_count    = 0
        if not self.api_key:
            logger.warning('GROQ_API_KEY not set. Using rule-based fallback.')

    @property
    def llm_available(self): return bool(self.api_key)

    def plan(self, context, global_state, patient, step):
        if step % self.plan_interval != 0 and self.last_plan is not None:
            return self.last_plan
        if self.api_key:
            try:
                p = _call_groq(context, self.api_key, self.model)
                p['source'] = 'groq'
                self.last_plan = p
                self.call_count += 1
                return p
            except urllib.error.HTTPError as e:
                if e.code == 429: time.sleep(5)
                elif e.code == 401: self.api_key = None
                self.fail_count += 1
            except Exception as e:
                self.fail_count += 1
                logger.warning(f'Groq failed: {e}')
        p = _fallback(global_state, patient)
        self.last_plan = p
        return p

    def encode_hint(self, plan):
        """4-float hint: [recommended_action_norm, risk_enc, transfer_flag, confidence]"""
        risk_map = {'low': 0.0, 'moderate': 0.33, 'high': 0.67, 'critical': 1.0}
        act      = float(plan.get('recommended_action', 0)) / 2.0
        risk     = risk_map.get(plan.get('risk_level', 'moderate'), 0.33)
        transfer = float(plan.get('allow_transfer', False))
        conf     = float(plan.get('confidence', 0.75))
        return [act, risk, transfer, conf]

    @property
    def stats(self):
        return {'backend': 'groq' if self.api_key else 'fallback',
                'model': self.model,
                'calls': self.call_count,
                'failures': self.fail_count}
