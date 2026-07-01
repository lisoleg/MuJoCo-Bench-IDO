"""
Standalone PD Standing Controller Verification Test
=====================================================

Tests that the PD controller keeps the dm_control humanoid standing upright
for at least 5 simulated seconds without falling.

This script replicates the same PD controller logic used in webviz/server.py
and verifies:
1. The robot maintains root_z > 1.0 (standing height) throughout the test.
2. No simulation crashes or unrecoverable instabilities.
3. Periodic motion (sinusoidal joint offsets) does not destabilize the robot.

Usage:
  C:\\Users\\1\\AppData\\Local\\Programs\\Python\\Python310\\python.exe test_pd_standing.py

Author: MuJoCo-Bench-IDO v0.4.2 verification
"""

import sys
import numpy as np


def main() -> None:
    """Run PD standing controller verification test."""
    print("=" * 60)
    print("  PD Standing Controller Verification Test")
    print("  MuJoCo-Bench-IDO v0.4.2")
    print("=" * 60)
    print()

    # ── Load dm_control humanoid-stand ──
    try:
        import dm_control.suite as suite
        import mujoco as mj
    except ImportError as e:
        print(f"ERROR: Required package not installed: {e}")
        print("Please install dm_control and mujoco.")
        sys.exit(1)

    print("[1] Loading dm_control humanoid-stand environment...")
    env = suite.load("humanoid", "stand")
    env.reset()

    mj_model = env.physics.model._model
    mj_data = env.physics.data._data

    print(f"    Model: nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    print(f"    Initial root_z = {mj_data.qpos[2]:.4f}")
    print()

    # ── Capture initial joint positions ──
    target_qpos = mj_data.qpos.copy()
    print(f"[2] Captured target_qpos (len={len(target_qpos)})")
    print(f"    Target root_z = {target_qpos[2]:.4f}")
    print()

    # ── Build actuator → (qpos_adr, dof_adr) mapping ──
    actuator_to_qpos_dof = []
    for i in range(mj_model.nu):
        jnt_id = int(mj_model.actuator_trnid[i, 0])
        qpos_adr = int(mj_model.jnt_qposadr[jnt_id])
        dof_adr = int(mj_model.jnt_dofadr[jnt_id])
        actuator_to_qpos_dof.append((qpos_adr, dof_adr))

    print(f"[3] Built actuator mapping ({len(actuator_to_qpos_dof)} actuators)")
    for i, (qpos_adr, dof_adr) in enumerate(actuator_to_qpos_dof):
        jnt_id = int(mj_model.actuator_trnid[i, 0])
        jnt_name_bytes = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        jnt_name = jnt_name_bytes.decode('utf-8') if isinstance(jnt_name_bytes, bytes) else str(jnt_name_bytes)
        print(f"    Actuator {i}: joint={jnt_name}, qpos_adr={qpos_adr}, dof_adr={dof_adr}")
    print()

    # ── Build motion joints mapping ──
    desired_motion = {
        "abdomen_z": (0.1, 0.5),
        "right_shoulder1": (0.15, 0.5),
        "left_shoulder1": (0.15, 0.5),
        "right_hip_x": (0.08, 0.3),
        "left_hip_x": (0.08, 0.3),
    }

    motion_joints: dict = {}
    for jnt_idx in range(mj_model.njnt):
        jnt_name_bytes = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, jnt_idx)
        if jnt_name_bytes is not None:
            jnt_name = jnt_name_bytes.decode('utf-8') if isinstance(jnt_name_bytes, bytes) else str(jnt_name_bytes)
            if jnt_name in desired_motion:
                qpos_adr = int(mj_model.jnt_qposadr[jnt_idx])
                dof_adr = int(mj_model.jnt_dofadr[jnt_idx])
                amp, freq = desired_motion[jnt_name]
                motion_joints[jnt_name] = (qpos_adr, dof_adr, amp, freq)

    print(f"[4] Motion joints configured ({len(motion_joints)} joints)")
    for jnt_name, (qpos_adr, dof_adr, amp, freq) in motion_joints.items():
        print(f"    {jnt_name}: amplitude={amp} rad, frequency={freq} Hz")
    print()

    # ── PD Controller Parameters ──
    KP: float = 50.0
    KD: float = 15.0
    print(f"[5] PD controller gains: KP={KP}, KD={KD}")
    print()

    # ── Run simulation for 5 seconds ──
    TEST_DURATION: float = 5.0  # seconds
    min_root_z: float = float('inf')
    max_root_z: float = 0.0
    step_count: int = 0
    warning_count: int = 0

    print(f"[6] Running simulation for {TEST_DURATION} seconds...")
    print(f"    sim_time | root_z  | min_root_z | status")
    print(f"    ---------|---------|------------|-------")

    last_report_time: float = 0.0
    root_z_history: list = []

    while mj_data.time < TEST_DURATION:
        # Clear applied forces
        mj_data.qfrc_applied[:] = 0.0

        # ── PD controller ──
        for i, (qpos_adr, dof_adr) in enumerate(actuator_to_qpos_dof):
            base_target = target_qpos[qpos_adr]

            # Add periodic motion offset for selected joints
            for jnt_name, (mqpos_adr, mdof_adr, amp, freq) in motion_joints.items():
                if qpos_adr == mqpos_adr:
                    offset = amp * np.sin(2.0 * np.pi * freq * mj_data.time)
                    base_target = base_target + offset
                    break

            error = base_target - mj_data.qpos[qpos_adr]
            vel = mj_data.qvel[dof_adr]
            torque = KP * error - KD * vel
            mj_data.qfrc_applied[dof_adr] = torque

        # Step physics
        mj.mj_step(mj_model, mj_data)
        step_count += 1

        # Record root_z
        root_z = float(mj_data.qpos[2])
        root_z_history.append(root_z)
        min_root_z = min(min_root_z, root_z)
        max_root_z = max(max_root_z, root_z)

        # Check for MuJoCo warnings
        if mj_data.warning[0].number > 0:
            warning_count += 1

        # Print status every ~1 second
        current_time = float(mj_data.time)
        if current_time - last_report_time >= 1.0 or current_time < 0.01:
            status = "STANDING" if root_z > 1.0 else "FALLING"
            print(f"    {current_time:7.3f}s | {root_z:7.4f} | {min_root_z:10.4f} | {status}")
            last_report_time = current_time

    # ── Results ──
    final_root_z = float(mj_data.qpos[2])
    avg_root_z = float(np.mean(root_z_history))

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Simulation duration: {mj_data.time:.2f} seconds ({step_count} steps)")
    print(f"  Final root_z:        {final_root_z:.4f}")
    print(f"  Average root_z:      {avg_root_z:.4f}")
    print(f"  Min root_z:          {min_root_z:.4f}")
    print(f"  Max root_z:          {max_root_z:.4f}")
    print(f"  MuJoCo warnings:     {warning_count}")
    print()

    # ── Verification ──
    PASS_THRESHOLD: float = 1.0
    if min_root_z > PASS_THRESHOLD:
        print(f"  VERIFICATION: PASS (min_root_z = {min_root_z:.4f} > {PASS_THRESHOLD})")
        print("  The PD controller successfully keeps the humanoid standing.")
    else:
        print(f"  VERIFICATION: FAIL (min_root_z = {min_root_z:.4f} <= {PASS_THRESHOLD})")
        print("  The robot fell during the test.")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
