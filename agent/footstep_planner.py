"""
Footstep Planner — Footstep Trajectory Planner with ZMP Validation
====================================================================

v0.17.0: World-frame footstep planning with SupportPolygon ZMP checking.

This module implements a footstep planner that generates physically-
feasible footstep sequences for bipedal/quadrupedal locomotion in
MuJoCo. It integrates with the psi-Anchor system for safety checking
and the kappa-Snap system for audit trail generation.

Key Components:
  - SupportPolygon: Computes support polygon from foot positions,
                    checks ZMP containment.
  - Footstep: Dataclass for a single footstep.
  - FootstepPlan: Complete plan with ordered footsteps.
  - FootstepPlanner: Main planner class that generates footstep
                     sequences from start to goal.

Algorithm:
  1. Decompose world-frame goal into direction + distance
  2. Generate nominal step length based on nominal gait parameters
  3. For each step:
     a. Compute desired foot landing position
     b. Check ZMP containment in support polygon
     c. If unsafe, adjust step length/width
     d. Compute swing trajectory (bezier curve)
     e. psi-Anchor audit
  4. Return FootstepPlan with all footsteps + safety flags

Author: MuJoCo-Bench-IDO v0.17.0
"""

import numpy as np
import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FootSide(Enum):
    """Which foot is stepping."""
    LEFT = "left"
    RIGHT = "right"


class StepPhase(Enum):
    """Phase of a single step."""
    DOUBLE_SUPPORT = "double_support"
    SINGLE_SUPPORT = "single_support"
    SWING = "swing"
    LANDING = "landing"


@dataclass
class Footstep:
    """A single footstep in world frame.

    Attributes:
        side: Left or right foot.
        position: [x, y, z] landing position in world frame.
        orientation: Yaw angle (radians).
        step_length: Distance from previous footstep of same side.
        step_width: Lateral distance from opposite foot.
        step_height: Vertical lift during swing (m).
        duration: Expected duration of this step (seconds).
        phase: Current phase of this step.
        zmp_safe: Whether ZMP was within support polygon at planning time.
        psi_violations: List of psi-Anchor violations (if any).
        ic: Information Cardinality of this step.
    """
    side: FootSide
    position: np.ndarray  # [x, y, z]
    orientation: float = 0.0
    step_length: float = 0.0
    step_width: float = 0.0
    step_height: float = 0.0
    duration: float = 0.5
    phase: StepPhase = StepPhase.SINGLE_SUPPORT
    zmp_safe: bool = True
    psi_violations: List[str] = field(default_factory=list)
    ic: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side.value,
            "position": self.position.tolist(),
            "orientation": round(self.orientation, 4),
            "step_length": round(self.step_length, 4),
            "step_width": round(self.step_width, 4),
            "step_height": round(self.step_height, 4),
            "duration": round(self.duration, 4),
            "phase": self.phase.value,
            "zmp_safe": self.zmp_safe,
            "psi_violations": self.psi_violations,
            "ic": round(self.ic, 6),
        }


@dataclass
class FootstepPlan:
    """A complete footstep plan from start to goal.

    Attributes:
        footsteps: Ordered list of footsteps.
        total_distance: Total path distance (m).
        total_steps: Number of footsteps.
        start_position: Starting position [x, y, z].
        goal_position: Goal position [x, y, z].
        avg_step_length: Average step length.
        zmp_violations: Number of steps with ZMP violations.
        psi_violations: Total psi-Anchor violations across all steps.
        plan_safe: Whether the entire plan is safe.
    """
    footsteps: List[Footstep] = field(default_factory=list)
    total_distance: float = 0.0
    total_steps: int = 0
    start_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    goal_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    avg_step_length: float = 0.0
    zmp_violations: int = 0
    psi_violations: int = 0
    plan_safe: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_distance": round(self.total_distance, 4),
            "total_steps": self.total_steps,
            "start_position": self.start_position.tolist(),
            "goal_position": self.goal_position.tolist(),
            "avg_step_length": round(self.avg_step_length, 4),
            "zmp_violations": self.zmp_violations,
            "psi_violations": self.psi_violations,
            "plan_safe": self.plan_safe,
            "footsteps": [fs.to_dict() for fs in self.footsteps],
        }


class SupportPolygon:
    """Support polygon for ZMP stability checking.

    The support polygon is the convex hull of all foot contact points.
    ZMP (Zero Moment Point) must remain inside this polygon for the
    robot to maintain balance.

    Usage:
        poly = SupportPolygon()
        poly.set_foot_positions(left_pos, right_pos)
        is_safe = poly.check_zmp(zmp_point)

    Attributes:
        left_foot_pos: Current left foot position [x, y].
        right_foot_pos: Current right foot position [x, y].
        foot_radius: Effective foot radius for contact area (m).
        safety_margin: Safety margin for ZMP containment (m).
    """

    def __init__(
        self,
        foot_radius: float = 0.03,
        safety_margin: float = 0.015,
    ) -> None:
        """Initialize support polygon.

        Args:
            foot_radius: Effective foot contact radius (m).
            safety_margin: ZMP safety margin from polygon edge (m).
        """
        self.foot_radius = foot_radius
        self.safety_margin = safety_margin
        self.left_foot_pos: np.ndarray = np.array([0.0, 0.05])
        self.right_foot_pos: np.ndarray = np.array([0.0, -0.05])
        self._vertices: List[np.ndarray] = []

    def set_foot_positions(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> None:
        """Set current foot positions and recompute polygon.

        Args:
            left: Left foot [x, y] position.
            right: Right foot [x, y] position.
        """
        self.left_foot_pos = np.array(left[:2], dtype=np.float64)
        self.right_foot_pos = np.array(right[:2], dtype=np.float64)
        self._compute_polygon()

    def _compute_polygon(self) -> None:
        """Compute the support polygon vertices.

        For two feet, the polygon is a rectangle with rounded corners
        (approximated by 8 vertices: 4 per foot circle).
        """
        vertices = []
        for foot_pos in [self.left_foot_pos, self.right_foot_pos]:
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                v = foot_pos + self.foot_radius * np.array([
                    math.cos(angle), math.sin(angle)
                ])
                vertices.append(v)

        # Compute convex hull (simplified: for 2 feet, just use the outermost vertices)
        self._vertices = self._convex_hull(vertices)

    @staticmethod
    def _convex_hull(points: List[np.ndarray]) -> List[np.ndarray]:
        """Compute convex hull using Andrew's monotone chain algorithm."""
        if len(points) <= 1:
            return points

        pts = sorted(points, key=lambda p: (p[0], p[1]))

        # Build lower hull
        lower = []
        for p in pts:
            while len(lower) >= 2 and SupportPolygon._cross(
                lower[-2], lower[-1], p
            ) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and SupportPolygon._cross(
                upper[-2], upper[-1], p
            ) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    @staticmethod
    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """2D cross product of vectors (a-o) and (b-o)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def check_zmp(self, zmp: np.ndarray) -> bool:
        """Check if ZMP is within the support polygon (with safety margin).

        Uses the cross-product test: ZMP is inside if it's on the
        same side of all edges.

        Args:
            zmp: ZMP point [x, y].

        Returns:
            True if ZMP is safely inside the polygon.
        """
        if len(self._vertices) < 3:
            return False

        zmp = np.array(zmp[:2], dtype=np.float64)

        # Check all edges
        n = len(self._vertices)
        for i in range(n):
            v1 = self._vertices[i]
            v2 = self._vertices[(i + 1) % n]

            # Compute edge normal (pointing inward)
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm

            # Distance from ZMP to edge
            dist = np.dot(zmp - v1, normal)

            # Must be inside (positive distance) with safety margin
            if dist < self.safety_margin:
                return False

        return True

    def get_polygon_area(self) -> float:
        """Compute support polygon area using the shoelace formula."""
        if len(self._vertices) < 3:
            return 0.0

        area = 0.0
        n = len(self._vertices)
        for i in range(n):
            j = (i + 1) % n
            area += self._vertices[i][0] * self._vertices[j][1]
            area -= self._vertices[j][0] * self._vertices[i][1]

        return abs(area) / 2.0

    def get_vertices(self) -> List[np.ndarray]:
        """Get the support polygon vertices."""
        return list(self._vertices)


class FootstepPlanner:
    """Footstep trajectory planner with ZMP validation.

    Generates physically-feasible footstep sequences from a start
    position to a goal position in world frame. Each step is validated
    against ZMP containment and psi-Anchor safety constraints.

    Algorithm:
      1. Compute direction vector from start to goal
      2. Determine number of steps based on nominal step length
      3. For each step:
         a. Compute desired foot landing position
         b. Update support polygon
         c. Estimate ZMP from CoM projection
         d. Check ZMP containment
         e. If unsafe, reduce step length and retry
         f. Compute swing trajectory parameters
         g. Compute Information Cardinality (IC)
      4. Return complete FootstepPlan

    Attributes:
        nominal_step_length: Default step length (m).
        nominal_step_width: Default step width (m).
        nominal_step_height: Default swing height (m).
        max_step_length: Maximum allowed step length (m).
        max_step_height: Maximum allowed swing height (m).
        step_duration: Default step duration (seconds).
        zmp_safety_margin: ZMP safety margin (m).
        support_polygon: SupportPolygon instance for ZMP checking.
    """

    def __init__(
        self,
        nominal_step_length: float = 0.08,
        nominal_step_width: float = 0.10,
        nominal_step_height: float = 0.02,
        max_step_length: float = 0.12,
        max_step_height: float = 0.03,
        step_duration: float = 0.5,
        zmp_safety_margin: float = 0.015,
        foot_radius: float = 0.03,
    ) -> None:
        """Initialize the footstep planner.

        Args:
            nominal_step_length: Default forward step length (m).
            nominal_step_width: Default lateral step width (m).
            nominal_step_height: Default vertical swing height (m).
            max_step_length: Maximum allowed step length (m).
            max_step_height: Maximum allowed swing height (m).
            step_duration: Default duration per step (seconds).
            zmp_safety_margin: Safety margin for ZMP containment (m).
            foot_radius: Effective foot contact radius (m).
        """
        self.nominal_step_length = nominal_step_length
        self.nominal_step_width = nominal_step_width
        self.nominal_step_height = nominal_step_height
        self.max_step_length = max_step_length
        self.max_step_height = max_step_height
        self.step_duration = step_duration
        self.zmp_safety_margin = zmp_safety_margin
        self.support_polygon = SupportPolygon(
            foot_radius=foot_radius,
            safety_margin=zmp_safety_margin,
        )

    def plan(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        start_left_foot: Optional[np.ndarray] = None,
        start_right_foot: Optional[np.ndarray] = None,
        max_steps: int = 50,
        start_side: FootSide = FootSide.LEFT,
    ) -> FootstepPlan:
        """Plan a footstep sequence from start to goal.

        Args:
            start_pos: Start position [x, y, z] in world frame.
            goal_pos: Goal position [x, y, z] in world frame.
            start_left_foot: Initial left foot position. If None, uses
                             start_pos + nominal_step_width/2 in y.
            start_right_foot: Initial right foot position. If None, uses
                              start_pos - nominal_step_width/2 in y.
            max_steps: Maximum number of footsteps to generate.
            start_side: Which foot steps first.

        Returns:
            FootstepPlan with ordered footsteps.
        """
        start_pos = np.array(start_pos, dtype=np.float64)
        goal_pos = np.array(goal_pos, dtype=np.float64)

        # Default foot positions
        if start_left_foot is None:
            start_left_foot = start_pos.copy()
            start_left_foot[1] += self.nominal_step_width / 2
        if start_right_foot is None:
            start_right_foot = start_pos.copy()
            start_right_foot[1] -= self.nominal_step_width / 2

        start_left_foot = np.array(start_left_foot, dtype=np.float64)
        start_right_foot = np.array(start_right_foot, dtype=np.float64)

        # Compute direction and distance
        direction_2d = goal_pos[:2] - start_pos[:2]
        total_distance = float(np.linalg.norm(direction_2d))

        if total_distance < 1e-6:
            return FootstepPlan(
                footsteps=[],
                total_distance=0.0,
                total_steps=0,
                start_position=start_pos,
                goal_position=goal_pos,
            )

        direction_2d = direction_2d / total_distance
        yaw = math.atan2(direction_2d[1], direction_2d[0])

        # Determine number of steps
        n_steps = min(
            max_steps,
            max(1, int(math.ceil(total_distance / self.nominal_step_length))),
        )

        # Adjust step length for even distribution
        actual_step_length = total_distance / n_steps
        if actual_step_length > self.max_step_length:
            actual_step_length = self.max_step_length
            n_steps = max(1, int(math.ceil(total_distance / actual_step_length)))

        # Initialize
        footsteps: List[Footstep] = []
        current_side = start_side
        left_pos = start_left_foot.copy()
        right_pos = start_right_foot.copy()
        zmp_violations = 0
        psi_violations_total = 0

        # Set initial support polygon
        self.support_polygon.set_foot_positions(left_pos, right_pos)

        for i in range(n_steps):
            # Compute desired landing position
            progress = (i + 1) / n_steps
            step_distance = actual_step_length * progress

            # Nominal landing position
            if current_side == FootSide.LEFT:
                swing_pos = left_pos.copy()
                swing_pos[0] = start_pos[0] + direction_2d[0] * step_distance
                swing_pos[1] = start_pos[1] + direction_2d[1] * step_distance + self.nominal_step_width / 2
                swing_pos[2] = start_pos[2]
                opposite_pos = right_pos
            else:
                swing_pos = right_pos.copy()
                swing_pos[0] = start_pos[0] + direction_2d[0] * step_distance
                swing_pos[1] = start_pos[1] + direction_2d[1] * step_distance - self.nominal_step_width / 2
                swing_pos[2] = start_pos[2]
                opposite_pos = left_pos

            # Compute step metrics
            if current_side == FootSide.LEFT:
                step_length = float(np.linalg.norm(swing_pos[:2] - left_pos[:2]))
                step_width = float(abs(swing_pos[1] - right_pos[1]))
            else:
                step_length = float(np.linalg.norm(swing_pos[:2] - right_pos[:2]))
                step_width = float(abs(swing_pos[1] - left_pos[1]))

            step_height = self.nominal_step_height

            # Estimate ZMP (simplified: CoM projection at mid-step)
            mid_pos = (swing_pos[:2] + opposite_pos[:2]) / 2
            estimated_zmp = mid_pos.copy()

            # Update support polygon (swing foot is still on ground at planning)
            self.support_polygon.set_foot_positions(left_pos, right_pos)

            # Check ZMP containment
            zmp_safe = self.support_polygon.check_zmp(estimated_zmp)

            # If unsafe, try reducing step length
            if not zmp_safe:
                # Reduce step length by 30%
                reduced_length = actual_step_length * 0.7
                direction_step = direction_2d * reduced_length
                if current_side == FootSide.LEFT:
                    swing_pos[0] = left_pos[0] + direction_step[0]
                    swing_pos[1] = left_pos[1] + direction_step[1] + self.nominal_step_width / 2
                else:
                    swing_pos[0] = right_pos[0] + direction_step[0]
                    swing_pos[1] = right_pos[1] + direction_step[1] - self.nominal_step_width / 2

                step_length = reduced_length

                # Recheck ZMP
                mid_pos = (swing_pos[:2] + opposite_pos[:2]) / 2
                estimated_zmp = mid_pos.copy()
                zmp_safe = self.support_polygon.check_zmp(estimated_zmp)

                if not zmp_safe:
                    zmp_violations += 1

            # Compute IC (Information Cardinality)
            # IC = position_change_entropy + velocity_variance
            position_delta = swing_pos[:2] - (left_pos[:2] if current_side == FootSide.LEFT else right_pos[:2])
            position_entropy = float(np.std(position_delta)) if len(position_delta) > 1 else float(abs(position_delta[0]))
            velocity_estimate = position_delta / self.step_duration
            velocity_var = float(np.var(velocity_estimate))
            ic = position_entropy + velocity_var

            # psi-Anchor checks (simplified)
            psi_violations: List[str] = []
            if step_length > self.max_step_length:
                psi_violations.append("MAX_STEP_LENGTH")
            if step_height > self.max_step_height:
                psi_violations.append("MAX_STEP_HEIGHT")
            if not zmp_safe:
                psi_violations.append("ZMP_VIOLATION")

            psi_violations_total += len(psi_violations)

            # Create footstep
            footstep = Footstep(
                side=current_side,
                position=swing_pos,
                orientation=yaw,
                step_length=step_length,
                step_width=step_width,
                step_height=step_height,
                duration=self.step_duration,
                phase=StepPhase.SINGLE_SUPPORT,
                zmp_safe=zmp_safe,
                psi_violations=psi_violations,
                ic=ic,
            )
            footsteps.append(footstep)

            # Update foot position
            if current_side == FootSide.LEFT:
                left_pos = swing_pos.copy()
            else:
                right_pos = swing_pos.copy()

            # Alternate sides
            current_side = FootSide.RIGHT if current_side == FootSide.LEFT else FootSide.LEFT

        # Build plan
        plan = FootstepPlan(
            footsteps=footsteps,
            total_distance=total_distance,
            total_steps=len(footsteps),
            start_position=start_pos,
            goal_position=goal_pos,
            avg_step_length=float(np.mean([fs.step_length for fs in footsteps])) if footsteps else 0.0,
            zmp_violations=zmp_violations,
            psi_violations=psi_violations_total,
            plan_safe=zmp_violations == 0 and psi_violations_total == 0,
        )

        return plan

    def compute_swing_trajectory(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        max_height: float,
        num_points: int = 50,
    ) -> np.ndarray:
        """Compute swing foot trajectory using a bezier curve.

        The trajectory starts at start_pos, arcs up to max_height at
        the midpoint, and lands at end_pos.

        Args:
            start_pos: Start position [x, y, z].
            end_pos: End position [x, y, z].
            max_height: Maximum swing height above ground (m).
            num_points: Number of trajectory points.

        Returns:
            Array of shape (num_points, 3) with trajectory points.
        """
        t = np.linspace(0, 1, num_points)

        # Quadratic bezier with height control
        # P(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
        # P1 is the apex point
        apex = (start_pos + end_pos) / 2
        apex[2] += max_height

        trajectory = np.zeros((num_points, 3))
        for i, ti in enumerate(t):
            for j in range(3):
                trajectory[i, j] = (
                    (1 - ti) ** 2 * start_pos[j]
                    + 2 * (1 - ti) * ti * apex[j]
                    + ti ** 2 * end_pos[j]
                )

        return trajectory

    def compute_com_trajectory(
        self,
        plan: FootstepPlan,
        num_points_per_step: int = 20,
    ) -> np.ndarray:
        """Compute Center of Mass trajectory for a footstep plan.

        The CoM follows a smooth trajectory that keeps the ZMP
        within the support polygon during each step.

        Args:
            plan: FootstepPlan to compute CoM for.
            num_points_per_step: Trajectory points per step.

        Returns:
            Array of shape (N, 3) with CoM positions.
        """
        if not plan.footsteps:
            return np.zeros((0, 3))

        all_points: List[np.ndarray] = []
        prev_left = plan.start_position.copy()
        prev_left[1] += self.nominal_step_width / 2
        prev_right = plan.start_position.copy()
        prev_right[1] -= self.nominal_step_width / 2

        for fs in plan.footsteps:
            if fs.side == FootSide.LEFT:
                start_com = (prev_left + prev_right) / 2
                end_com = (fs.position + prev_right) / 2
                prev_left = fs.position.copy()
            else:
                start_com = (prev_left + prev_right) / 2
                end_com = (prev_left + fs.position) / 2
                prev_right = fs.position.copy()

            # Linear interpolation for CoM (simplified)
            for i in range(num_points_per_step):
                t = i / num_points_per_step
                com = start_com * (1 - t) + end_com * t
                # Add slight height variation
                com[2] += 0.01 * math.sin(math.pi * t)
                all_points.append(com)

        return np.array(all_points)

    def plan_with_obstacle_avoidance(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacles: List[Dict[str, Any]],
        **kwargs,
    ) -> FootstepPlan:
        """Plan footstep sequence with obstacle avoidance.

        Uses a simplified potential field approach: obstacles create
        repulsive potentials that bend the path around them.

        Args:
            start_pos: Start position [x, y, z].
            goal_pos: Goal position [x, y, z].
            obstacles: List of obstacle dicts with 'position' and 'radius'.
            **kwargs: Additional arguments passed to plan().

        Returns:
            FootstepPlan that avoids obstacles.
        """
        start_pos = np.array(start_pos, dtype=np.float64)
        goal_pos = np.array(goal_pos, dtype=np.float64)

        if not obstacles:
            return self.plan(start_pos, goal_pos, **kwargs)

        # Compute waypoints using potential field
        direction = goal_pos[:2] - start_pos[:2]
        total_dist = float(np.linalg.norm(direction))
        if total_dist < 1e-6:
            return self.plan(start_pos, goal_pos, **kwargs)

        direction = direction / total_dist
        n_waypoints = max(3, int(total_dist / 0.3))
        waypoints = [start_pos.copy()]

        for i in range(1, n_waypoints):
            t = i / n_waypoints
            nominal_pos = start_pos[:2] + t * (goal_pos[:2] - start_pos[:2])

            # Apply repulsive potential from obstacles
            repulsion = np.zeros(2)
            for obs in obstacles:
                obs_pos = np.array(obs.get("position", [0, 0])[:2], dtype=np.float64)
                obs_radius = float(obs.get("radius", 0.1))
                diff = nominal_pos - obs_pos
                dist = float(np.linalg.norm(diff))
                if dist < obs_radius * 2:
                    if dist < 1e-6:
                        dist = 1e-6
                    repulsion += (diff / dist) * (obs_radius * 2 - dist) * 0.5

            adjusted_pos = nominal_pos + repulsion
            wp = np.array([adjusted_pos[0], adjusted_pos[1], start_pos[2]])
            waypoints.append(wp)

        waypoints.append(goal_pos.copy())

        # Plan through waypoints
        all_footsteps: List[Footstep] = []
        total_distance = 0.0
        zmp_violations = 0
        psi_violations_total = 0

        for i in range(len(waypoints) - 1):
            segment_plan = self.plan(
                waypoints[i],
                waypoints[i + 1],
                start_side=FootSide.LEFT if i % 2 == 0 else FootSide.RIGHT,
                **kwargs,
            )
            all_footsteps.extend(segment_plan.footsteps)
            total_distance += segment_plan.total_distance
            zmp_violations += segment_plan.zmp_violations
            psi_violations_total += segment_plan.psi_violations

        return FootstepPlan(
            footsteps=all_footsteps,
            total_distance=total_distance,
            total_steps=len(all_footsteps),
            start_position=start_pos,
            goal_position=goal_pos,
            avg_step_length=float(np.mean([fs.step_length for fs in all_footsteps])) if all_footsteps else 0.0,
            zmp_violations=zmp_violations,
            psi_violations=psi_violations_total,
            plan_safe=zmp_violations == 0 and psi_violations_total == 0,
        )
