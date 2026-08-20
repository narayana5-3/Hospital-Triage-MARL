import json, os
from datetime import datetime
from collections import defaultdict


class Message:
    def __init__(self, sender, msg_type, payload=None):
        self.sender    = sender
        self.msg_type  = msg_type
        self.payload   = payload or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {'sender': self.sender, 'type': self.msg_type,
                'payload': self.payload, 'ts': self.timestamp}


class MessageBus:
    """
    Inter-ward communication channel.

    Message types:
      CAPACITY_ALERT      - ward occupancy >= 80%
      RESOURCE_OFFER      - ward occupancy <= 40%
      ESCALATION_NEEDED   - General patient deteriorated, needs ICU
      DEESCALATION_READY  - ICU patient improved, ready for General
      TRIAGE_OVERFLOW     - Emergency cannot accept new patient
      CRITICAL_UNPLACED   - Critical patient has no ICU bed
    """

    MSG_TYPES = [
        'CAPACITY_ALERT', 'RESOURCE_OFFER',
        'ESCALATION_NEEDED', 'DEESCALATION_READY',
        'TRIAGE_OVERFLOW', 'CRITICAL_UNPLACED',
    ]

    def __init__(self, log_dir='logs'):
        self._queue = []
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, 'message_bus.jsonl')

    def post(self, sender, msg_type, payload=None):
        msg = Message(sender, msg_type, payload)
        self._queue.append(msg)
        return msg

    def read(self, msg_type=None, sender=None):
        msgs = self._queue
        if msg_type: msgs = [m for m in msgs if m.msg_type == msg_type]
        if sender:   msgs = [m for m in msgs if m.sender   == sender]
        return msgs

    def flush(self):
        for m in self._queue:
            with open(self._log_path, 'a') as f:
                f.write(json.dumps(m.to_dict()) + '\n')
        self._queue = []

    def summarise(self):
        counts = defaultdict(int)
        for m in self._queue:
            counts[m.msg_type] += 1
        return dict(counts)

    def pending_dicts(self):
        return [m.to_dict() for m in self._queue]

    def encode_hint(self):
        """
        4-float encoding of bus state appended to each ward agent observation.
        [capacity_pressure, escalation_pressure, deescalation_pressure, critical_pressure]
        """
        cap   = min(1.0, sum(1 for m in self._queue
                             if m.msg_type == 'CAPACITY_ALERT')   / 3.0)
        esc   = min(1.0, sum(1 for m in self._queue
                             if m.msg_type == 'ESCALATION_NEEDED')/ 3.0)
        deesc = min(1.0, sum(1 for m in self._queue
                             if m.msg_type == 'DEESCALATION_READY')/ 3.0)
        crit  = min(1.0, sum(1 for m in self._queue
                             if m.msg_type == 'CRITICAL_UNPLACED') / 3.0)
        return [cap, esc, deesc, crit]
