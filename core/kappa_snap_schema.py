"""
κ-Snap JSON Schema — 20 Event Type Definitions & Validation
=============================================================

Defines all 20 κ-Snap event types with JSON Schema validation
for the Machine Conscience Audit Framework (v0.6.0).

Each κ-Snap event is a structured JSON dict validated against
the schema before being logged to the MerkleChain.

Event types span 7 audit levels (L0–L6):
  L0=System, L1=Noether, L2=Psi, L3=PGate, L4=Adaptation, L5=Task, L6=Meta

Author: MuJoCo-Bench-IDO v0.6.0 — Machine Conscience Audit Framework
"""

import json
import time
from typing import Any, Dict, List, Optional

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

IDO_KAPPA_SNAP_SCHEMA_VERSION: str = "v0.1.0"


# ── 20 Event Type Definitions ──

EVENT_TYPES: Dict[str, Dict[str, Any]] = {
    "INIT": {
        "level": "L0",
        "description": "Episode initialization",
        "required_details": ["task_name", "goal_delta_K"],
    },
    "ACTION_ACCEPT": {
        "level": "L0",
        "description": "PG-Gate passed — action accepted",
        "required_details": ["action_norm", "tau_safe"],
    },
    "REJECT_FRICTION_CONE": {
        "level": "L1",
        "description": "Friction cone violation: ||f_t|| > μ·f_n",
        "required_details": ["f_t", "f_n", "mu", "contact_id"],
    },
    "REJECT_ENERGY_VIOLATION": {
        "level": "L1",
        "description": "Energy budget exceeded: ΔE > budget",
        "required_details": ["delta_E", "budget"],
    },
    "REJECT_SENTIENT_LIMIT": {
        "level": "L2",
        "description": "Sentient torque limit exceeded: τ > τ_sentient_max",
        "required_details": ["joint_name", "torque", "tau_max"],
    },
    "REJECT_SELF_COLLISION": {
        "level": "L1",
        "description": "Self-collision detected: min_dist < thresh",
        "required_details": ["min_dist", "thresh", "body_pair"],
    },
    "REJECT_PG_GATE": {
        "level": "L3",
        "description": "PG-Gate rejected action (AST/physical clamp)",
        "required_details": ["ast_reason", "original_action", "clamped_action"],
    },
    "CREATIVE_PROBE": {
        "level": "L4",
        "description": "Creative-Probe triggered due to η stagnation",
        "required_details": ["probe_type", "probe_params"],
    },
    "THERMAL_DRIFT": {
        "level": "L4",
        "description": "η gradual increase (thermal drift)",
        "required_details": ["drift_rate", "baseline_eta"],
    },
    "SCREW_LOOSENING": {
        "level": "L4",
        "description": "Parameter drift detected",
        "required_details": ["param_name", "expected", "actual"],
    },
    "CALIBRATION_DRIFT": {
        "level": "L4",
        "description": "Sensor calibration offset detected",
        "required_details": ["sensor_id", "drift_amount"],
    },
    "SENSOR_DEGRADED": {
        "level": "L4",
        "description": "Sensor degradation detected",
        "required_details": ["sensor_id", "degradation_pct"],
    },
    "SELF_REFLECT": {
        "level": "L6",
        "description": "ψ-Anchor evolution triggered",
        "required_details": ["evo_policy", "epiplexity", "conservation_score"],
    },
    "FINGER_TORQUE_CLAMPED": {
        "level": "L2",
        "description": "Finger torque clamped to safe limit",
        "required_details": ["joint_name", "original_torque", "clamped_torque"],
    },
    "WIND_GUST": {
        "level": "L5",
        "description": "Wind gust disturbance detected",
        "required_details": ["wind_speed", "wind_direction"],
    },
    "BIOMASS_DETECTED": {
        "level": "L5",
        "description": "Biomass/sentient body proximity detected",
        "required_details": ["body_id", "proximity"],
    },
    "TASK_START": {
        "level": "L5",
        "description": "Task execution started",
        "required_details": ["task_name"],
    },
    "TASK_COMPLETE": {
        "level": "L5",
        "description": "Task execution completed",
        "required_details": ["task_name", "final_eta", "total_steps"],
    },
    "SAFE_STOP": {
        "level": "L0",
        "description": "L4 fuse triggered — safe stop",
        "required_details": ["fuse_level", "trigger_reason"],
    },
    "FATAL_ERROR": {
        "level": "L0",
        "description": "System fatal error",
        "required_details": ["error_type", "error_msg"],
    },
}


# ── JSON Schema Definition ──

KAPPA_SNAP_EVENT_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "KappaSnapEvent",
    "description": "κ-Snap audit event schema for Machine Conscience Framework",
    "type": "object",
    "required": ["event_type", "level", "eta", "decision", "snap_id",
                 "prev_snap_id", "timestamp", "details"],
    "properties": {
        "event_type": {
            "type": "string",
            "enum": list(EVENT_TYPES.keys()),
            "description": "One of the 20 κ-Snap event types",
        },
        "level": {
            "type": "string",
            "enum": ["L0", "L1", "L2", "L3", "L4", "L5", "L6"],
            "description": "Audit level classification",
        },
        "eta": {
            "type": "number",
            "description": "Current κ-Snap residual η value",
        },
        "decision": {
            "type": "string",
            "description": "Decision string (e.g., 'EXPLOIT', 'SAFE', 'ACCEPT')",
        },
        "snap_id": {
            "type": "string",
            "description": "MerkleChain snap ID (prev_snap_id + sha256[:16])",
        },
        "prev_snap_id": {
            "type": "string",
            "description": "Previous MerkleChain snap ID for chain linkage",
        },
        "timestamp": {
            "type": "number",
            "description": "Unix timestamp of event creation",
        },
        "details": {
            "type": "object",
            "description": "Event-specific details dict",
            "additionalProperties": True,
        },
    },
    "additionalProperties": False,
}


class KappaSnapSchema:
    """κ-Snap JSON Schema validator for Machine Conscience Audit events.

    Validates that every κ-Snap event dict conforms to the defined schema,
    ensuring event_type is one of the 20 defined types, level matches the
    expected level for that event type, and required detail fields are present.

    Attributes:
        VERSION: Schema version string.
        EVENT_TYPES: Dict of all 20 event type definitions.
    """

    VERSION: str = IDO_KAPPA_SNAP_SCHEMA_VERSION

    def __init__(self) -> None:
        """Initialize KappaSnapSchema validator."""
        self._event_types: Dict[str, Dict[str, Any]] = EVENT_TYPES
        self._schema: Dict[str, Any] = KAPPA_SNAP_EVENT_SCHEMA

    def validate(self, event_dict: Dict[str, Any]) -> bool:
        """Validate an event dict against the κ-Snap JSON Schema.

        Checks:
        1. All required fields are present (event_type, level, eta, decision,
           snap_id, prev_snap_id, timestamp, details).
        2. event_type is one of the 20 defined types.
        3. level matches the expected level for the event_type.
        4. Required detail fields for the event_type are present.

        If jsonschema library is available, also validates against the full
        JSON Schema definition. Otherwise, uses manual validation.

        Args:
            event_dict: Event dict to validate.

        Returns:
            True if the event dict is valid, False otherwise.
        """
        # Check required fields
        required_fields: List[str] = [
            "event_type", "level", "eta", "decision",
            "snap_id", "prev_snap_id", "timestamp", "details",
        ]
        for field in required_fields:
            if field not in event_dict:
                return False

        # Check event_type is valid
        event_type: str = event_dict.get("event_type", "")
        if event_type not in self._event_types:
            return False

        # Check level matches expected level for event_type
        expected_level: str = self._event_types[event_type]["level"]
        actual_level: str = event_dict.get("level", "")
        if actual_level != expected_level:
            return False

        # Check required detail fields for event_type
        required_details: List[str] = self._event_types[event_type].get(
            "required_details", [])
        details: Dict[str, Any] = event_dict.get("details", {})
        for detail_field in required_details:
            if detail_field not in details:
                return False

        # If jsonschema is available, validate against full schema
        if HAS_JSONSCHEMA:
            try:
                jsonschema.validate(instance=event_dict, schema=self._schema)
            except jsonschema.ValidationError:
                return False

        return True

    def create_event(self,
                     event_type: str,
                     level: Optional[str] = None,
                     eta: float = 0.0,
                     decision: str = "",
                     snap_id: str = "",
                     prev_snap_id: str = "",
                     details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a κ-Snap event dict with proper format.

        Automatically fills in the expected level for the given event_type
        if level is not explicitly provided.

        Args:
            event_type: One of the 20 κ-Snap event types.
            level: Audit level (L0–L6). If None, auto-filled from event_type.
            eta: Current κ-Snap residual η value.
            decision: Decision string for this step.
            snap_id: MerkleChain snap ID.
            prev_snap_id: Previous MerkleChain snap ID.
            details: Event-specific details dict.

        Returns:
            Event dict conforming to κ-Snap JSON Schema.

        Raises:
            ValueError: If event_type is not one of the 20 defined types.
        """
        if event_type not in self._event_types:
            raise ValueError(
                f"Unknown event_type '{event_type}'. "
                f"Must be one of: {list(self._event_types.keys())}")

        # Auto-fill level from event_type definition
        if level is None:
            level = self._event_types[event_type]["level"]

        # Default details to empty dict
        if details is None:
            details = {}

        event: Dict[str, Any] = {
            "event_type": event_type,
            "level": level,
            "eta": eta,
            "decision": decision,
            "snap_id": snap_id,
            "prev_snap_id": prev_snap_id,
            "timestamp": time.time(),
            "details": details,
        }

        return event

    def get_schema(self) -> Dict[str, Any]:
        """Return the full κ-Snap JSON Schema definition.

        Returns:
            Dict representing the JSON Schema for κ-Snap events.
        """
        return self._schema

    def get_event_types(self) -> Dict[str, Dict[str, Any]]:
        """Return all 20 κ-Snap event type definitions.

        Returns:
            Dict mapping event_type name → definition dict.
        """
        return self._event_types
