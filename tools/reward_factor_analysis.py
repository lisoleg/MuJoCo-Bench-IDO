"""Humanoid-stand reward factor analysis - decompose dm_control reward per step."""
import sys
import numpy as np
sys.path.insert(0, '.')

import dm_control.suite as suite


def tolerance(x, bounds, margin=0.0, value_at_margin=0.0, sigmoid='gaussian'):
    """dm_control tolerance function."""
    lower, upper = bounds
    if lower <= x <= upper:
        return 1.0
    if margin == 0:
        return 0.0
    excess = max(abs(x - lower), abs(x - upper))
    if sigmoid == 'quadratic':
        return max(0.0, value_at_margin + (1 - value_at_margin) * (1 - (excess / margin) ** 2))
    elif sigmoid == 'linear':
        return max(0.0, 1 - excess / margin)
    elif sigmoid == 'gaussian':
        return max(0.0, np.exp(-((excess / margin) ** 2)))
    return 0.0


def run_humanoid_stand_analysis(episode_steps=500):
    """Run humanoid-stand with IDO agent and log reward factors."""
    env = suite.load(domain_name='humanoid', task_name='stand')
    timestep = env.reset()

    from agent.mujoco_ido_agent import IDOMuJoCoAgent
    from core.goal_eml_mj import make_humanoid_stand_eml
    goal = make_humanoid_stand_eml(env.physics, delta_K=0.05)
    agent = IDOMuJoCoAgent(env, goal, task_name='humanoid-stand',
                            kappa_thresh=0.05, enable_critique=True)

    total_reward = 0.0
    step_rewards = []
    small_controls = []
    stand_rewards_list = []
    dont_moves_list = []
    standing_vals = []
    upright_vals = []
    head_heights = []

    for step_idx in range(episode_steps):
        action = agent.choose_action(timestep, physics=env.physics)
        timestep = env.step(action)

        step_reward = float(timestep.reward or 0.0)
        total_reward += step_reward
        step_rewards.append(step_reward)

        phys = env.physics

        # Head height
        head_z = float(phys.named.data.xpos['head', 2])
        head_heights.append(head_z)

        # Standing: tolerance(head_height, bounds=(1.4, inf), margin=0.35)
        standing = tolerance(head_z, bounds=(1.4, float('inf')), margin=0.35)
        standing_vals.append(standing)

        # Torso upright (xmat zz component)
        torso_upright = float(phys.named.data.xmat['torso', :][8])
        upright_vals.append(torso_upright)

        # stand_reward = standing * upright
        # dm_control uses upright = (1 + torso_upright) / 2 for humanoid-stand
        upright_reward = (1 + torso_upright) / 2
        stand_reward_val = standing * upright_reward
        stand_rewards_list.append(stand_reward_val)

        # small_control per-actuator: tolerance(ctrl, margin=1, sigmoid='quadratic').mean()
        # Then: (4 + mean_tolerance) / 5
        sc_per_actuator = []
        for c in action:
            sc_per_actuator.append(tolerance(abs(c), bounds=(0, 0), margin=1.0, value_at_margin=0.0, sigmoid='quadratic'))
        small_control_raw = float(np.mean(sc_per_actuator))
        small_control_val = (4 + small_control_raw) / 5
        small_controls.append(small_control_val)

        # dont_move: tolerance(horizontal_velocity, margin=2).mean()
        try:
            torso_vel = phys.named.data.sensordata['torso_subtreelinvel']
            horiz_vel_x = float(torso_vel[0])
            horiz_vel_y = float(torso_vel[1])
        except (KeyError, IndexError, TypeError):
            horiz_vel_x = 0.0
            horiz_vel_y = 0.0

        dont_move_x = tolerance(abs(horiz_vel_x), bounds=(0, 0), margin=2.0)
        dont_move_y = tolerance(abs(horiz_vel_y), bounds=(0, 0), margin=2.0)
        dont_move_val = (dont_move_x + dont_move_y) / 2
        dont_moves_list.append(dont_move_val)

        if timestep.last():
            print(f"Episode ended at step {step_idx+1}")
            break

    n = len(step_rewards)
    print(f"\n{'='*60}")
    print(f"  HUMANOID-STAND REWARD FACTOR ANALYSIS (IDO agent)")
    print(f"  Steps: {n} | Total reward: {total_reward:.4f}")
    print(f"  Avg per-step reward: {total_reward/n:.6f}")
    print(f"{'='*60}")

    print(f"\n  --- Reward Factor Breakdown ---")
    print(f"  small_control:   mean={np.mean(small_controls):.4f}  min={np.min(small_controls):.4f}  max={np.max(small_controls):.4f}")
    print(f"  standing:        mean={np.mean(standing_vals):.4f}  min={np.min(standing_vals):.4f}  max={np.max(standing_vals):.4f}")
    print(f"  torso_upright:   mean={np.mean(upright_vals):.4f}  min={np.min(upright_vals):.4f}  max={np.max(upright_vals):.4f}")
    print(f"  stand_reward:    mean={np.mean(stand_rewards_list):.4f}  min={np.min(stand_rewards_list):.4f}  max={np.max(stand_rewards_list):.4f}")
    print(f"  dont_move:       mean={np.mean(dont_moves_list):.4f}  min={np.min(dont_moves_list):.4f}  max={np.max(dont_moves_list):.4f}")

    print(f"\n  --- Expected vs Actual ---")
    expected_reward = np.mean(small_controls) * np.mean(stand_rewards_list) * np.mean(dont_moves_list)
    print(f"  Expected avg reward (product of means): {expected_reward:.6f}")
    print(f"  Actual avg reward: {np.mean(step_rewards):.6f}")

    print(f"\n  --- Head Height ---")
    print(f"  head_z:          mean={np.mean(head_heights):.4f}  min={np.min(head_heights):.4f}  max={np.max(head_heights):.4f}")
    above_14 = sum(1 for h in head_heights if h >= 1.4)
    print(f"  head_z >= 1.4:   {above_14}/{n} = {above_14/n:.2%}")

    # Factor contribution analysis
    print(f"\n  --- Reward Factor Impact (what-if) ---")
    sc_m = np.mean(small_controls)
    sr_m = np.mean(stand_rewards_list)
    dm_m = np.mean(dont_moves_list)
    print(f"  If small_control=1.0:  total ~ {1.0 * sr_m * dm_m * n:.2f}")
    print(f"  If stand_reward=1.0:  total ~ {sc_m * 1.0 * dm_m * n:.2f}")
    print(f"  If dont_move=1.0:     total ~ {sc_m * sr_m * 1.0 * n:.2f}")
    print(f"  If ALL=1.0 (perfect): total ~ {n} = {n}")

    print(f"\n  --- Gap vs dm_control baseline (~800) ---")
    print(f"  Current total: {total_reward:.4f}")
    print(f"  Gap factor: {800/max(total_reward, 0.001):.1f}x")
    print(f"  Per-step needed: 0.8 | Per-step actual: {total_reward/n:.6f}")

    # BOTTLENECK identification
    print(f"\n  --- BOTTLENECK IDENTIFICATION ---")
    factors = {'small_control': sc_m, 'stand_reward': sr_m, 'dont_move': dm_m}
    bottleneck = min(factors, key=factors.get)
    print(f"  Lowest factor: {bottleneck} = {factors[bottleneck]:.4f}")
    print(f"  This is the primary bottleneck limiting reward!")

    # Step reward distribution
    print(f"\n  --- Step Reward Distribution ---")
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    for i in range(len(bins)-1):
        count = sum(1 for r in step_rewards if bins[i] <= r < bins[i+1])
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} steps ({count/n:.1%})")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    run_humanoid_stand_analysis()
