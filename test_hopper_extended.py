"""Extended test: 10 episodes x 1000 steps, summary only."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from dm_control import suite
from agent.task_pd_controllers import HopperHopPD


def main():
    n_episodes = 10
    max_steps = 1000

    env = suite.load('hopper', 'hop')

    all_returns = []
    all_max_heights = []
    all_max_speeds = []
    all_standing_steps = []
    all_hopping_steps = []
    all_positive_reward_steps = []

    for ep in range(n_episodes):
        timestep = env.reset()
        controller = HopperHopPD(env.physics)

        total_reward = 0.0
        max_height = 0.0
        max_speed = 0.0
        standing_steps = 0
        hopping_steps = 0
        positive_reward_steps = 0
        ep_heights = []

        for step in range(max_steps):
            if timestep.last():
                break

            ctrl = controller.compute_action(timestep, env.physics)
            height = float(env.physics.height())
            speed = float(env.physics.speed())
            reward = timestep.reward or 0.0

            max_height = max(max_height, height)
            max_speed = max(max_speed, abs(speed))
            ep_heights.append(height)
            total_reward += reward

            if height > 0.6:
                standing_steps += 1
            if abs(speed) > 2.0:
                hopping_steps += 1
            if reward > 0:
                positive_reward_steps += 1

            timestep = env.step(ctrl)

        rooty_init = float(env.physics.data.qpos[2])  # not available after reset loop
        all_returns.append(total_reward)
        all_max_heights.append(max_height)
        all_max_speeds.append(max_speed)
        all_standing_steps.append(standing_steps)
        all_hopping_steps.append(hopping_steps)
        all_positive_reward_steps.append(positive_reward_steps)

        print(f"Ep {ep+1:2d}: return={total_reward:8.4f} max_h={max_height:.3f} "
              f"max_s={max_speed:.3f} standing={standing_steps:4d} hopping={hopping_steps:4d} "
              f"pos_reward={positive_reward_steps:4d}")

    avg_return = np.mean(all_returns)
    avg_max_h = np.mean(all_max_heights)
    avg_max_s = np.mean(all_max_speeds)
    total_standing = sum(all_standing_steps)
    total_hopping = sum(all_hopping_steps)
    total_positive = sum(all_positive_reward_steps)

    print(f"\n=== SUMMARY ({n_episodes} eps x {max_steps} steps) ===")
    print(f"  Avg return: {avg_return:.4f}")
    print(f"  Avg max height: {avg_max_h:.4f}")
    print(f"  Avg max speed: {avg_max_s:.4f}")
    print(f"  Total standing steps: {total_standing}/{n_episodes*max_steps}")
    print(f"  Total hopping steps: {total_hopping}/{n_episodes*max_steps}")
    print(f"  Total positive reward steps: {total_positive}/{n_episodes*max_steps}")
    print(f"  Per-ep returns: {[f'{r:.4f}' for r in all_returns]}")

    return avg_return


if __name__ == '__main__':
    main()
