"""
KappaSnapTokenizer — κ-Snap Audit Trail → VLA/LLM Special Token Encoding
=======================================================================

v0.16.25 P0: KappaSnapTokenizer

Encodes the κ-Snap audit trail (step-level causal snapshots) as special
tokens that can be prepended to VLA/LLM input context. This gives the
VLA model access to its own causal history — violations, η trajectory,
decision modes — so it can make more informed, safety-aware decisions.

Design:
  Each κ-Snap event is encoded as a compact token:
    [KSNAP:<level>:<event_short>:<eta_bucket>:<decision_short>]

  Example:
    [KSNAP:L0:ACTACC:lo:EXP]   — Action accepted, low η, EXPLOIT mode
    [KSNAP:L1:FRIC:hi:SAFE]    — Friction cone violation, high η, SAFE mode
    [KSNAP:L2:SENT:mid:EXP]    — Sentient limit hit, mid η, EXPLOIT mode

  The tokenizer maintains a sliding window of recent events and produces:
    1. A token string for LLM text input
    2. A token ID list for embedding layer input
    3. A compact vector summary for non-LLM VLA models

Integration:
  - VLA adapter's obs_dict gets a 'kappa_tokens' key
  - S-Bridge's ask_why_llm() uses tokens as context for LLM attribution
  - T-Processor's κ-FIFO feeds tokens to the η-ALU

Author: MuJoCo-Bench-IDO v0.16.25 — P0 Feature
"""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

# ── Token vocabulary ──

# Event type short codes (maps full event type → 6-char code)
EVENT_SHORT_CODES: Dict[str, str] = {
    "INIT": "INIT",
    "ACTION_ACCEPT": "ACTACC",
    "REJECT_FRICTION_CONE": "FRIC",
    "REJECT_ENERGY_VIOLATION": "ENERG",
    "REJECT_SENTIENT_LIMIT": "SENT",
    "REJECT_SELF_COLLISION": "COLL",
    "REJECT_PG_GATE": "PGATE",
    "CREATIVE_PROBE": "PROBE",
    "THERMAL_DRIFT": "THRM",
    "FUSE_WARNING": "FUSEW",
    "FUSE_INFO": "FUSEI",
    "FUSE_BLOCK": "FUSEB",
    "PRE_AFFECT_SIGNAL": "AFFECT",
    "EVIDENCE_CHECK": "EVID",
    "TASK_SUCCESS": "SUCC",
    "TASK_FAIL": "FAIL",
    "ETA_DESCEND": "ETAD",
    "ETA_PLATEAU": "ETAP",
    "ETA_ASCEND": "ETAA",
    "META_EVOLVE": "EVOLVE",
}

# Decision mode short codes
DECISION_SHORT_CODES: Dict[str, str] = {
    "EXPLOIT": "EXP",
    "EXPLORE": "EXR",
    "SAFE": "SAFE",
    "CREATIVE_PROBE": "CPRB",
    "FUSE_BLOCK": "FBLK",
    "FUSE_WARNING": "FWARN",
    "FUSE_INFO": "FINFO",
    "NORMAL": "NORM",
}

# η bucket boundaries
ETA_BUCKETS: List[Tuple[float, str]] = [
    (0.01, "vlo"),   # very low: η < 0.01
    (0.1, "lo"),     # low: 0.01 ≤ η < 0.1
    (0.5, "mid"),    # mid: 0.1 ≤ η < 0.5
    (2.0, "hi"),     # high: 0.5 ≤ η < 2.0
    (float("inf"), "vhi"),  # very high: η ≥ 2.0
]

# Audit level short codes (already short, just uppercase)
LEVEL_CODES: Dict[str, str] = {
    "L0": "L0", "L1": "L1", "L2": "L2",
    "L3": "L3", "L4": "L4", "L5": "L5", "L6": "L6",
}

# Special tokens
KSNAP_START_TOKEN: str = "[KSNAP_START]"
KSNAP_END_TOKEN: str = "[KSNAP_END]"
KSNAP_PAD_TOKEN: str = "[KSNAP_PAD]"
KSNAP_SEP_TOKEN: str = "[KSNAP_SEP]"

# Vocabulary size (for embedding layer)
VOCAB_SIZE: int = 256  # Enough for all combinations + special tokens


@dataclass
class KappaSnapToken:
    """A single encoded κ-Snap token.

    Attributes:
        raw: Raw token string, e.g. "[KSNAP:L0:ACTACC:lo:EXP]"
        event_type: Original event type name.
        level: Audit level (L0-L6).
        eta_bucket: η bucket code (vlo/lo/mid/hi/vhi).
        decision: Decision short code.
        token_id: Integer token ID for embedding.
        timestamp: Simulation time when event occurred.
        snap_id: κ-Snap ID from Merkle chain.
    """
    raw: str = ""
    event_type: str = ""
    level: str = "L0"
    eta_bucket: str = "mid"
    decision: str = "NORM"
    token_id: int = 0
    timestamp: float = 0.0
    snap_id: str = ""


def _eta_to_bucket(eta: float) -> str:
    """Map η value to bucket code."""
    abs_eta = abs(eta)
    for threshold, code in ETA_BUCKETS:
        if abs_eta < threshold:
            return code
    return "vhi"


def _event_to_short(event_type: str) -> str:
    """Map event type to 6-char short code."""
    return EVENT_SHORT_CODES.get(event_type, event_type[:6].upper())


def _decision_to_short(decision: str) -> str:
    """Map decision string to short code."""
    if decision in DECISION_SHORT_CODES:
        return DECISION_SHORT_CODES[decision]
    # Try prefix matching for compound decisions like "EXPLOIT/SAFE"
    for key, code in DECISION_SHORT_CODES.items():
        if key in decision.upper():
            return code
    return decision[:4].upper() if decision else "NORM"


def _token_to_id(level: str, event_short: str, eta_bucket: str, decision_short: str) -> int:
    """Compute a deterministic token ID from token components.

    Uses a simple hash to map (level, event, eta, decision) → 0..VOCAB_SIZE-1.
    """
    raw = f"{level}:{event_short}:{eta_bucket}:{decision_short}"
    h = int(hashlib.md5(raw.encode("utf-8")).hexdigest(), 16)
    return h % VOCAB_SIZE


class KappaSnapTokenizer:
    """κ-Snap audit trail → special token encoder for VLA/LLM context.

    Maintains a sliding window of recent κ-Snap events and encodes them
    as special tokens. These tokens can be:
      1. Prepended to VLA observation as text (for LLM-based VLAs)
      2. Converted to token IDs for embedding layer input
      3. Summarized as a compact vector for non-LLM VLAs

    Usage:
        tokenizer = KappaSnapTokenizer(window_size=16)

        # Feed events from KappaSnapLogger
        for event in logger.get_recent_events(n=16):
            tokenizer.add_event(event)

        # Get token string for LLM context
        token_str = tokenizer.get_token_string()
        # → "[KSNAP_START] [KSNAP:L0:ACTACC:lo:EXP] [KSNAP:L1:FRIC:hi:SAFE] ... [KSNAP_END]"

        # Get token IDs for embedding
        token_ids = tokenizer.get_token_ids()
        # → [1, 42, 128, 1, ...]

        # Get compact vector summary
        summary_vec = tokenizer.get_summary_vector()
        # → np.ndarray of shape (32,)

    Attributes:
        window_size: Maximum number of recent events to encode.
        _buffer: Sliding window of KappaSnapToken objects.
    """

    VERSION: str = "v0.16.25"

    def __init__(self, window_size: int = 16) -> None:
        """Initialize tokenizer with sliding window.

        Args:
            window_size: Maximum number of recent κ-Snap events to keep.
                        Older events are dropped (FIFO). Default 16.
        """
        self.window_size: int = window_size
        self._buffer: deque = deque(maxlen=window_size)

        # Token ID mapping (lazy-filled)
        self._id_map: Dict[str, int] = {}
        self._next_id: int = 5  # Reserve 0-4 for special tokens

        # Pre-register special tokens
        self._id_map[KSNAP_PAD_TOKEN] = 0
        self._id_map[KSNAP_START_TOKEN] = 1
        self._id_map[KSNAP_END_TOKEN] = 2
        self._id_map[KSNAP_SEP_TOKEN] = 3

    def add_event(self, event: Dict[str, Any]) -> KappaSnapToken:
        """Add a κ-Snap event to the sliding window and encode it.

        Args:
            event: κ-Snap event dict from KappaSnapLogger. Expected keys:
                   - 'event_type' (str): Event type name
                   - 'level' (str): Audit level (L0-L6)
                   - 'eta' (float): η residual value
                   - 'decision' (str): Decision mode string
                   - 'timestamp' (float, optional): Sim time
                   - 'snap_id' (str, optional): Merkle chain snap ID

        Returns:
            KappaSnapToken with encoded information.
        """
        event_type = event.get("event_type", "ACTION_ACCEPT")
        level = event.get("level", "L0")
        eta = float(event.get("eta", 0.0))
        decision = event.get("decision", "NORMAL")
        timestamp = float(event.get("timestamp", 0.0))
        snap_id = event.get("snap_id", "")

        event_short = _event_to_short(event_type)
        eta_bucket = _eta_to_bucket(eta)
        decision_short = _decision_to_short(decision)

        # Build token string
        raw = f"[KSNAP:{level}:{event_short}:{eta_bucket}:{decision_short}]"

        # Compute token ID
        token_id = _token_to_id(level, event_short, eta_bucket, decision_short)

        token = KappaSnapToken(
            raw=raw,
            event_type=event_type,
            level=level,
            eta_bucket=eta_bucket,
            decision=decision_short,
            token_id=token_id,
            timestamp=timestamp,
            snap_id=snap_id,
        )

        self._buffer.append(token)
        return token

    def add_events(self, events: List[Dict[str, Any]]) -> List[KappaSnapToken]:
        """Batch add multiple κ-Snap events.

        Args:
            events: List of κ-Snap event dicts.

        Returns:
            List of encoded KappaSnapToken objects.
        """
        return [self.add_event(e) for e in events]

    def get_token_string(self) -> str:
        """Get the encoded token string for LLM text context.

        Returns:
            Token string like:
            "[KSNAP_START] [KSNAP:L0:ACTACC:lo:EXP] [KSNAP:L1:FRIC:hi:SAFE] [KSNAP_END]"
        """
        if not self._buffer:
            return f"{KSNAP_START_TOKEN} {KSNAP_END_TOKEN}"

        tokens = [t.raw for t in self._buffer]
        return f"{KSNAP_START_TOKEN} {' '.join(tokens)} {KSNAP_END_TOKEN}"

    def get_token_ids(self, max_len: Optional[int] = None) -> List[int]:
        """Get token IDs for embedding layer input.

        Args:
            max_len: Maximum sequence length. If None, uses window_size + 2
                    (for START/END tokens). Padded with PAD token ID (0).

        Returns:
            List of integer token IDs.
        """
        if max_len is None:
            max_len = self.window_size + 2

        ids = [self._id_map[KSNAP_START_TOKEN]]
        for token in self._buffer:
            ids.append(token.token_id)
        ids.append(self._id_map[KSNAP_END_TOKEN])

        # Pad to max_len
        while len(ids) < max_len:
            ids.append(self._id_map[KSNAP_PAD_TOKEN])

        # Truncate to max_len
        return ids[:max_len]

    def get_summary_vector(self, dim: int = 32) -> "np.ndarray":
        """Get a compact vector summary of the κ-Snap history.

        Encodes the distribution of event types, η buckets, and decision
        modes in the sliding window as a fixed-length vector. This is
        useful for non-LLM VLA models that can't process text tokens.

        Vector layout (dim=32):
          [0:5]   η bucket counts (vlo, lo, mid, hi, vhi) — normalized
          [5:12]  Level counts (L0-L6) — normalized
          [12:18] Decision mode counts (EXP, EXR, SAFE, CPRB, FBLK, NORM)
          [18:24] Violation indicator (1 if any L1/L2/L3 event in window)
          [24:28] Recent η trend (mean of last 4 η values, delta, min, max)
          [28:32] Violation rate, action accept rate, diversity, recency

        Args:
            dim: Output vector dimension. Default 32.

        Returns:
            np.ndarray of shape (dim,) float32.
        """
        import numpy as np

        vec = np.zeros(dim, dtype=np.float32)

        if not self._buffer:
            return vec

        tokens = list(self._buffer)
        n = len(tokens)

        # η bucket counts [0:5]
        bucket_names = ["vlo", "lo", "mid", "hi", "vhi"]
        bucket_counts = {b: 0 for b in bucket_names}
        for t in tokens:
            if t.eta_bucket in bucket_counts:
                bucket_counts[t.eta_bucket] += 1
        for i, b in enumerate(bucket_names):
            vec[i] = bucket_counts[b] / n if dim > i else 0.0

        # Level counts [5:12]
        level_names = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
        level_counts = {l: 0 for l in level_names}
        for t in tokens:
            if t.level in level_counts:
                level_counts[t.level] += 1
        for i, l in enumerate(level_names):
            idx = 5 + i
            if dim > idx:
                vec[idx] = level_counts[l] / n

        # Decision mode counts [12:18]
        decision_names = ["EXP", "EXR", "SAFE", "CPRB", "FBLK", "NORM"]
        decision_counts = {d: 0 for d in decision_names}
        for t in tokens:
            if t.decision in decision_counts:
                decision_counts[t.decision] += 1
        for i, d in enumerate(decision_names):
            idx = 12 + i
            if dim > idx:
                vec[idx] = decision_counts[d] / n

        # Violation indicator [18:24]
        violation_levels = {"L1", "L2", "L3"}
        has_violation = any(t.level in violation_levels for t in tokens)
        if dim > 18:
            vec[18] = 1.0 if has_violation else 0.0
        violation_count = sum(1 for t in tokens if t.level in violation_levels)
        if dim > 19:
            vec[19] = violation_count / n
        # Action accept rate
        accept_count = sum(1 for t in tokens if t.event_type == "ACTION_ACCEPT")
        if dim > 20:
            vec[20] = accept_count / n
        # Creative probe rate
        probe_count = sum(1 for t in tokens if t.event_type == "CREATIVE_PROBE")
        if dim > 21:
            vec[21] = probe_count / n

        # Recent η trend [24:28]
        recent_etas = []
        for t in tokens[-4:]:
            # Reverse-map bucket to representative η value
            eta_repr = {"vlo": 0.005, "lo": 0.05, "mid": 0.3, "hi": 1.0, "vhi": 3.0}
            recent_etas.append(eta_repr.get(t.eta_bucket, 0.3))
        if recent_etas:
            if dim > 24:
                vec[24] = float(np.mean(recent_etas))
            if dim > 25 and len(recent_etas) >= 2:
                vec[25] = recent_etas[-1] - recent_etas[0]
            if dim > 26:
                vec[26] = float(min(recent_etas))
            if dim > 27:
                vec[27] = float(max(recent_etas))

        # Aggregate stats [28:32]
        if dim > 28:
            vec[28] = violation_count / max(n, 1)  # violation rate
        if dim > 29:
            vec[29] = accept_count / max(n, 1)  # accept rate
        if dim > 30:
            # Diversity: unique event types / total
            unique_types = len(set(t.event_type for t in tokens))
            vec[30] = unique_types / max(n, 1)
        if dim > 31:
            # Recency weight: more recent events weighted higher
            recency = sum((i + 1) / n for i in range(n)) / n
            vec[31] = recency

        return vec

    def get_obs_dict_extras(self) -> Dict[str, Any]:
        """Get extras dict for VLA adapter obs_dict integration.

        Returns a dict with keys:
          - 'kappa_tokens': token string for LLM context
          - 'kappa_token_ids': token ID list for embedding
          - 'kappa_summary': compact vector summary

        Usage:
            obs_dict.update(tokenizer.get_obs_dict_extras())
        """
        return {
            'kappa_tokens': self.get_token_string(),
            'kappa_token_ids': self.get_token_ids(),
            'kappa_summary': self.get_summary_vector(),
        }

    def clear(self) -> None:
        """Clear the sliding window buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return number of tokens in the buffer."""
        return len(self._buffer)

    def get_recent_events_summary(self) -> str:
        """Get a human-readable summary of recent κ-Snap events.

        Returns:
            Multi-line string with recent event details.
        """
        if not self._buffer:
            return "No κ-Snap events in buffer."

        lines = [f"κ-Snap Tokenizer — {len(self._buffer)} events in window:"]
        for i, token in enumerate(list(self._buffer)[-8:]):  # Last 8
            lines.append(
                f"  [{i+1}] {token.raw}  "
                f"(snap={token.snap_id[:8]}, t={token.timestamp:.2f}s)"
            )
        return "\n".join(lines)
