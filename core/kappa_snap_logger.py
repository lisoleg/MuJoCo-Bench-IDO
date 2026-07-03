"""
κ-Snap Logger — L0~L6 Audit Level Classification + MerkleChain Integration
============================================================================

Provides categorized logging for the Machine Conscience Audit Framework (v0.6.0 → v0.8.0).

Each κ-Snap event is validated by KappaSnapSchema, then appended to the
embedded MerkleChain for tamper-proof audit trail. Events are classified
into 7 audit levels:

  L0=System, L1=Noether, L2=Psi, L3=PGate, L4=Adaptation, L5=Task, L6=Meta

The MerkleChain ensures immutability:
  snap_id = prev_snap_id + sha256(prev_snap_id + str(η) + str(decision))[:16]

v0.8.0 升级项 U3: 新增 log_to_jsonl() 方法
  - 调用 KappaSnapJSONLWriter 将事件写入 JSONL 文件
  - Hermes 翻译层处理私有标签 → 人可读映射

Author: MuJoCo-Bench-IDO v0.8.0 — Machine Conscience Audit Framework
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from core.kappa_snap_schema import KappaSnapSchema
from core.kappa_snap_jsonl import KappaSnapJSONLWriter, HermesTranslator

IDO_KAPPA_SNAP_LOGGER_VERSION: str = "v0.2.0"


# ── Audit Level Definitions ──

LEVELS: Dict[str, Dict[str, Any]] = {
    "L0": {"name": "System",    "description": "System-level events (INIT, SAFE_STOP, FATAL_ERROR)"},
    "L1": {"name": "Noether",   "description": "Noether conservation gate violations"},
    "L2": {"name": "Psi",       "description": "ψ-Anchor sentient limit checks"},
    "L3": {"name": "PGate",     "description": "PG-Gate hard anchor clamp events"},
    "L4": {"name": "Adaptation","description": "Adaptive behavior (Creative-Probe, drift detection)"},
    "L5": {"name": "Task",      "description": "Task-level events (TASK_START, WIND_GUST)"},
    "L6": {"name": "Meta",      "description": "Meta-management (ψ-Anchor evolution)"},
}


class MerkleChain:
    """Tamper-proof audit chain for κ-Snap events.

    Each append computes a snap_id linking to the previous entry via
    SHA-256 hashing, creating a Merkle-like chain where any tampering
    with a single entry breaks the chain integrity.

    Hash computation rule:
        snap_id = prev_snap_id + sha256(prev_snap_id + str(η) + str(decision))[:16]

    Attributes:
        VERSION: Chain version string.
    """

    VERSION: str = "v0.3.0"

    def __init__(self) -> None:
        """Initialize MerkleChain with genesis snap_id."""
        self._prev_snap_id: str = "genesis"
        self._chain: List[Dict[str, Any]] = []

    def append(self,
               eta: float,
               decision: str,
               event_type: str = "ACTION_ACCEPT",
               level: str = "L0") -> str:
        """Append a new entry to the MerkleChain.

        Computes snap_id from prev_snap_id + SHA-256 hash, stores
        the complete entry, and updates prev_snap_id for next link.

        Args:
            eta: κ-Snap residual η value for this step.
            decision: Decision string (e.g., 'EXPLOIT', 'SAFE', 'ACCEPT').
            event_type: κ-Snap event type (one of 20 defined types).
            level: Audit level (L0–L6).

        Returns:
            The computed snap_id for this entry.
        """
        # Compute snap_id using SHA-256 hash rule
        hash_input: str = self._prev_snap_id + str(eta) + str(decision)
        hash_hex: str = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        snap_id: str = self._prev_snap_id + hash_hex[:16]

        entry: Dict[str, Any] = {
            "snap_id": snap_id,
            "prev_snap_id": self._prev_snap_id,
            "eta": eta,
            "decision": decision,
            "event_type": event_type,
            "level": level,
            "hash": hash_hex[:16],
            "timestamp": time.time(),
        }

        self._chain.append(entry)
        self._prev_snap_id = snap_id

        return snap_id

    def verify(self) -> bool:
        """Verify the integrity of the entire MerkleChain.

        Checks that each entry's snap_id can be recomputed from
        prev_snap_id + sha256(prev_snap_id + str(η) + str(decision))[:16],
        and that prev_snap_id matches the previous entry's snap_id.

        Returns:
            True if the chain is intact (no tampering detected), False otherwise.
        """
        if len(self._chain) == 0:
            return True

        expected_prev: str = "genesis"

        for entry in self._chain:
            # Check prev_snap_id linkage
            if entry["prev_snap_id"] != expected_prev:
                return False

            # Recompute snap_id hash
            hash_input: str = entry["prev_snap_id"] + str(entry["eta"]) + str(entry["decision"])
            recomputed_hash: str = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
            expected_snap_id: str = entry["prev_snap_id"] + recomputed_hash

            # Check snap_id matches recomputed value
            if entry["snap_id"] != expected_snap_id:
                return False

            # Check stored hash matches recomputed hash
            if entry["hash"] != recomputed_hash:
                return False

            expected_prev = entry["snap_id"]

        return True

    def get_chain(self) -> List[Dict[str, Any]]:
        """Return the complete chain of entries.

        Returns:
            List of all MerkleChain entries in order.
        """
        return list(self._chain)

    def get_last_snap_id(self) -> str:
        """Return the last snap_id in the chain (for next entry linkage).

        Returns:
            Last snap_id string, or 'genesis' if chain is empty.
        """
        if len(self._chain) > 0:
            return self._chain[-1]["snap_id"]
        return self._prev_snap_id

    def reset(self) -> None:
        """Reset the chain to genesis state."""
        self._prev_snap_id = "genesis"
        self._chain = []

    def _compute_hash(self, snap_id: str, eta: float, decision: str) -> str:
        """Compute SHA-256 hash for a chain entry.

        Args:
            snap_id: Previous snap_id (used as hash input prefix).
            eta: η value for this step.
            decision: Decision string.

        Returns:
            First 16 characters of the SHA-256 hex digest.
        """
        hash_input: str = snap_id + str(eta) + str(decision)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]


class KappaSnapLogger:
    """κ-Snap audit event logger with MerkleChain integration.

    Logs κ-Snap events at 7 audit levels (L0–L6), validates them
    against KappaSnapSchema, and appends them to the embedded
    MerkleChain for tamper-proof audit trail.

    Usage:
        logger = KappaSnapLogger()
        event = logger.log("ACTION_ACCEPT", "L0", 0.05, "EXPLOIT", physics)
        logger.verify_chain()  # Verify Merkle integrity

    Attributes:
        VERSION: Logger version string.
        LEVELS: Dict of audit level definitions.
    """

    VERSION: str = IDO_KAPPA_SNAP_LOGGER_VERSION

    def __init__(self, schema: Optional[KappaSnapSchema] = None) -> None:
        """Initialize KappaSnapLogger with schema validator and MerkleChain.

        Args:
            schema: Optional KappaSnapSchema instance for validation.
                    If None, auto-created with default configuration.
        """
        self._schema: KappaSnapSchema = schema if schema is not None else KappaSnapSchema()
        self._merkle: MerkleChain = MerkleChain()
        self._log_buffer: List[Dict[str, Any]] = []
        # ── v0.8.0 升级项 U3: JSONL 步骤级审计输出 ──
        # 默认不启用 — 需显式调用 enable_jsonl() 或 log_to_jsonl()
        self._jsonl_writer: Optional[KappaSnapJSONLWriter] = None

    def log(self,
            event_type: str,
            level: str,
            eta: float,
            decision: str,
            physics: Optional[Any] = None,
            details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Log a κ-Snap audit event and append to MerkleChain.

        Creates a validated event dict via KappaSnapSchema, appends
        η and decision to the MerkleChain, and returns the complete
        event dict.

        Args:
            event_type: One of the 20 κ-Snap event types.
            level: Audit level (L0–L6).
            eta: Current κ-Snap residual η value.
            decision: Decision string for this step.
            physics: Optional MuJoCo physics data (for context extraction).
            details: Optional event-specific details dict.

        Returns:
            Validated event dict with snap_id, prev_snap_id, and Merkle linkage.
        """
        # Get Merkle linkage
        prev_snap_id: str = self._merkle.get_last_snap_id()

        # Create event via schema
        event: Dict[str, Any] = self._schema.create_event(
            event_type=event_type,
            level=level,
            eta=eta,
            decision=decision,
            snap_id="",  # Will be filled by MerkleChain append
            prev_snap_id=prev_snap_id,
            details=details or {},
        )

        # Append to MerkleChain (computes snap_id)
        snap_id: str = self._merkle.append(
            eta=eta,
            decision=decision,
            event_type=event_type,
            level=level,
        )

        # Update event with actual snap_id
        event["snap_id"] = snap_id
        event["prev_snap_id"] = prev_snap_id

        # Validate event against schema
        if not self._schema.validate(event):
            # Schema validation failed — still log but mark as unvalidated
            event["_validated"] = False
        else:
            event["_validated"] = True

        # Store in log buffer
        self._log_buffer.append(event)

        return event

    def get_merkle_chain(self) -> List[Dict[str, Any]]:
        """Return the complete MerkleChain.

        Returns:
            List of all MerkleChain entries in order.
        """
        return self._merkle.get_chain()

    def verify_chain(self) -> bool:
        """Verify the integrity of the MerkleChain.

        Returns:
            True if the chain is intact (no tampering detected).
        """
        return self._merkle.verify()

    def get_log_buffer(self) -> List[Dict[str, Any]]:
        """Return all logged events from the buffer.

        Returns:
            List of all event dicts logged since initialization or last reset.
        """
        return list(self._log_buffer)

    def enable_jsonl(self, file_path: str) -> None:
        """启用 JSONL 步骤级审计输出 — v0.8.0 升级项 U3.

        打开 JSONL 文件, 开始将每步事件写入 JSONL 文件.
        默认不启用 — 需显式调用此方法.

        Args:
            file_path: JSONL 文件路径 (如 "logs/kappa_snap_cheetah-run_ep0.jsonl").
        """
        self._jsonl_writer = KappaSnapJSONLWriter()
        self._jsonl_writer.open(file_path)

    def log_to_jsonl(self,
                     eta: float,
                     mode: str,
                     fuse_level: str,
                     pre_affect: str,
                     noether_result: Optional[Dict[str, Any]] = None,
                     evidence_verified: Optional[bool] = None) -> str:
        """将一步事件写入 JSONL 文件 — v0.8.0 升级项 U3.

        调用 KappaSnapJSONLWriter.write_step() 将当前步骤的审计
        记录写入 JSONL 文件, 包含 η, mode, fuse_level, pre_affect 等字段.

        若 JSONL 输出未启用 (未调用 enable_jsonl), 则仅返回空字符串.

        Args:
            eta: κ-Snap 残差 η 值.
            mode: Agent 模式 (EXPLOIT/EXPLORE/SAFE).
            fuse_level: SafeFuse 级别.
            pre_affect: PreAffect 信号 (GRRR/PHEW/NEUTRAL).
            noether_result: Noether 检查结果 Dict.
            evidence_verified: 证据校验标记 (可选).

        Returns:
            本步 snap_id 字符串. 若未启用 JSONL, 返回空字符串.
        """
        if self._jsonl_writer is None:
            return ""
        snap_id: str = self._jsonl_writer.write_step(
            eta=eta,
            mode=mode,
            fuse_level=fuse_level,
            pre_affect=pre_affect,
            noether_result=noether_result,
            evidence_verified=evidence_verified,
        )
        return snap_id

    def get_jsonl_writer(self) -> Optional[KappaSnapJSONLWriter]:
        """返回 JSONL 写入器实例 — v0.8.0 升级项 U3.

        Returns:
            KappaSnapJSONLWriter 实例, 或 None (未启用).
        """
        return self._jsonl_writer

    def reset(self) -> None:
        """Reset logger state (MerkleChain + log buffer) for a new episode."""
        self._merkle.reset()
        self._log_buffer = []
        # ── v0.8.0 升级项 U3: 重置 JSONL 写入器 ──
        if self._jsonl_writer is not None:
            self._jsonl_writer.reset()
            self._jsonl_writer = None

    def _format_event(self,
                      event_type: str,
                      level: str,
                      eta: float,
                      decision: str,
                      details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format a κ-Snap event dict without MerkleChain linkage.

        Used for events that don't need MerkleChain recording
        (e.g., debug/trace events).

        Args:
            event_type: κ-Snap event type string.
            level: Audit level string (L0–L6).
            eta: η value.
            decision: Decision string.
            details: Optional details dict.

        Returns:
            Formatted event dict (not appended to MerkleChain).
        """
        return {
            "event_type": event_type,
            "level": level,
            "eta": eta,
            "decision": decision,
            "timestamp": time.time(),
            "details": details or {},
        }
