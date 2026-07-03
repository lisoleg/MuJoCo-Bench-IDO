"""Diagnostic test to determine waist actuator sign convention for rooty control."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dm_control import suite


def test_waist_sign():
    """Test whether positive ctrl[0] makes rooty increase or decrease.

    Strategy: start near upright, apply constant positive ctrl[0],
    observe rooty change. Then apply constant negative ctrl[0].
    """
    env = suite.load('hopper', 'hop')

    # Test 1: Positive ctrl[0] = +0.5, all other ctrl = 0
    print("\n=== Test 1: ctrl[0]=+0.5, rest=0 ===")
    timestep = env.reset()
    rooty_initial = float(env.physics.data.qpos[2])
    print(f"  Initial rooty: {rooty_initial:.4f}")

    for step in range(20):
        ctrl = np.array([0.5, 0.0, 0.0, 0.0])
        rooty = float(env.physics.data.qpos[2])
        rootz = float(env.physics.data.qpos[1])
        height = float(env.physics.height())
        waist = float(env.physics.data.qpos[3])
        print(f"  step {step}: rooty={rooty:.4f} rootz={rootz:.4f} height={height:.4f} waist={waist:.4f}")
        timestep = env.step(ctrl)

    # Test 2: Negative ctrl[0] = -0.5, all other ctrl = 0
    print("\n=== Test 2: ctrl[0]=-0.5, rest=0 ===")
    timestep = env.reset()
    rooty_initial = float(env.physics.data.qpos[2])
    print(f"  Initial rooty: {rooty_initial:.4f}")

    for step in range(20):
        ctrl = np.array([-0.5, 0.0, 0.0, 0.0])
        rooty = float(env.physics.data.qpos[2])
        rootz = float(env.physics.data.qpos[1])
        height = float(env.physics.height())
        waist = float(env.physics.data.qpos[3])
        print(f"  step {step}: rooty={rooty:.4f} rootz={rootz:.4f} height={height:.4f} waist={waist:.4f}")
        timestep = env.step(ctrl)

    # Test 3: Positive ctrl[1] = +0.5 (hip), all other ctrl = 0
    print("\n=== Test 3: ctrl[1]=+0.5 (hip), rest=0 ===")
    timestep = env.reset()
    rooty_initial = float(env.physics.data.qpos[2])
    hip_initial = float(env.physics.data.qpos[4])
    print(f"  Initial rooty: {rooty_initial:.4f}, hip: {hip_initial:.4f}")

    for step in range(20):
        ctrl = np.array([0.0, 0.5, 0.0, 0.0])
        rooty = float(env.physics.data.qpos[2])
        rootz = float(env.physics.data.qpos[1])
        height = float(env.physics.height())
        hip = float(env.physics.data.qpos[4])
        print(f"  step {step}: rooty={rooty:.4f} rootz={rootz:.4f} height={height:.4f} hip={hip:.4f}")
        timestep = env.step(ctrl)

    # Test 4: Positive ctrl[2] = +0.5 (knee), all other ctrl = 0
    print("\n=== Test 4: ctrl[2]=+0.5 (knee), rest=0 ===")
    timestep = env.reset()
    rooty_initial = float(env.physics.data.qpos[2])
    knee_initial = float(env.physics.data.qpos[5])
    print(f"  Initial rooty: {rooty_initial:.4f}, knee: {knee_initial:.4f}")

    for step in range(20):
        ctrl = np.array([0.0, 0.0, 0.5, 0.0])
        rooty = float(env.physics.data.qpos[2])
        rootz = float(env.physics.data.qpos[1])
        height = float(env.physics.height())
        knee = float(env.physics.data.qpos[5])
        speed = float(env.physics.speed())
        print(f"  step {step}: rooty={rooty:.4f} rootz={rootz:.4f} height={height:.4f} knee={knee:.4f} speed={speed:.4f}")
        timestep = env.step(ctrl)


def test_all_actuators_extend():
    """Test the 'extend leg' strategy: what happens when we push all leg actuators."""
    env = suite.load('hopper', 'hop')

    print("\n=== Test 5: Leg extend (hip=0.5, knee=0.5, ankle=0.3, waist=0) ===")
    timestep = env.reset()
    rooty_initial = float(env.physics.data.qpos[2])
    print(f"  Initial rooty: {rooty_initial:.4f}")

    for step in range(50):
        ctrl = np.array([0.0, 0.5, 0.5, 0.3])
        rooty = float(env.physics.data.qpos[2])
        rootz = float(env.physics.data.qpos[1])
        height = float(env.physics.height())
        speed = float(env.physics.speed())
        reward = timestep.reward or 0.0
        touch = timestep.observation.get('touch', np.zeros(2))
        print(f"  step {step}: h={height:.4f} s={speed:.4f} r={reward:.4f} "
              f"rooty={rooty:.4f} rootz={rootz:.4f} touch=[{touch[0]:.2f},{touch[1]:.2f}]")
        timestep = env.step(ctrl)


if __name__ == '__main__':
    test_waist_sign()
    test_all_actuators_extend()
