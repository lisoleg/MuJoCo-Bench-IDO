"""Quick test for HopperHopPD controller — 3 episodes, reports reward/height/speed."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dm_control import suite

# Import the controller
from agent.task_pd_controllers import HopperHopPD


def run_episode(env, max_steps=200, verbose=False):
    """Run one episode and return total reward, heights, speeds."""
    timestep = env.reset()
    total_reward = 0.0
    heights = []
    speeds = []
    rootys = []
    rootz_list = []
    ctrls_list = []

    # Create controller with new physics
    controller = HopperHopPD(env.physics)

    for step in range(max_steps):
        if timestep.last():
            break

        ctrl = controller.compute_action(timestep, env.physics)
        ctrls_list.append(ctrl.copy())

        height = float(env.physics.height())
        speed = float(env.physics.speed())
        rooty = float(env.physics.data.qpos[2])
        rootz = float(env.physics.data.qpos[1])
        reward = timestep.reward or 0.0

        heights.append(height)
        speeds.append(speed)
        rootys.append(rooty)
        rootz_list.append(rootz)
        total_reward += reward

        if verbose and step < 30:
            touch = timestep.observation.get('touch', np.zeros(2))
            print(f"  step {step:3d}: h={height:.3f} s={speed:.3f} r={reward:.4f} "
                  f"rooty={rooty:.3f} rootz={rootz:.3f} ctrl=[{ctrl[0]:.2f},{ctrl[1]:.2f},{ctrl[2]:.2f},{ctrl[3]:.2f}] "
                  f"touch=[{touch[0]:.3f},{touch[1]:.3f}]")

        timestep = env.step(ctrl)

    return total_reward, heights, speeds, rootys, rootz_list, ctrls_list


def main():
    n_episodes = 5
    max_steps = 200

    env = suite.load('hopper', 'hop')

    all_returns = []
    all_heights = []
    all_speeds = []
    all_rootys = []

    for ep in range(n_episodes):
        print(f"\n=== Episode {ep+1}/{n_episodes} ===")
        ret, heights, speeds, rootys, rootz_list, ctrls_list = run_episode(
            env, max_steps=max_steps, verbose=True)

        avg_height = np.mean(heights) if heights else 0.0
        avg_speed = np.mean(speeds) if speeds else 0.0
        max_height = np.max(heights) if heights else 0.0
        max_speed = np.max(speeds) if speeds else 0.0
        avg_rooty = np.mean(rootys) if rootys else 0.0

        all_returns.append(ret)
        all_heights.extend(heights)
        all_speeds.extend(speeds)
        all_rootys.extend(rootys)

        print(f"  Return: {ret:.4f}")
        print(f"  Avg height: {avg_height:.4f}, Max height: {max_height:.4f}")
        print(f"  Avg speed: {avg_speed:.4f}, Max speed: {max_speed:.4f}")
        print(f"  Avg rooty: {avg_rooty:.4f}")
        # First 5 steps rooty and rootz
        print(f"  Initial rootys: {[f'{r:.2f}' for r in rootys[:5]]}")
        print(f"  Initial rootz: {[f'{z:.2f}' for z in rootz_list[:5]]}")

    avg_return = np.mean(all_returns)
    avg_height = np.mean(all_heights) if all_heights else 0.0
    avg_speed = np.mean(all_speeds) if all_speeds else 0.0

    print(f"\n=== SUMMARY ===")
    print(f"  Avg return: {avg_return:.4f}")
    print(f"  Avg height: {avg_height:.4f}")
    print(f"  Avg speed: {avg_speed:.4f}")
    print(f"  Per-ep returns: {[f'{r:.4f}' for r in all_returns]}")

    # Check how many steps had height > 0.6 (standing reward threshold)
    standing_steps = sum(1 for h in all_heights if h > 0.6)
    total_steps = len(all_heights)
    print(f"  Standing steps (h>0.6): {standing_steps}/{total_steps} = {standing_steps/total_steps*100:.1f}%")

    # Check how many steps had speed > 2.0 (hopping reward threshold)
    hopping_steps = sum(1 for s in all_speeds if s > 2.0)
    print(f"  Hopping steps (s>2.0): {hopping_steps}/{total_steps} = {hopping_steps/total_steps*100:.1f}%")

    # Check rooty distribution (how many start upside-down?)
    print(f"  |rooty| > 1.0 steps: {sum(1 for r in all_rootys if abs(r) > 1.0)}/{total_steps}")

    return avg_return


if __name__ == '__main__':
    main()
