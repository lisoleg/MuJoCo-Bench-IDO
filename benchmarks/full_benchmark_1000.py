"""
Full 1000-step benchmark with normalized score comparison to industry SOTA.
"""
import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dm_control.suite as suite
from stable_baselines3 import PPO, SAC
from baselines.sb3_adapter import SB3PPOAdapter, SB3SACAdapter
from core.goal_eml_mj import GoalEML
from agent.hybrid_sb3_ido_agent import HybridSB3IDOAgent
from agent.task_pd_controllers import get_controller_for_task
from benchmarks.run_mujoco_bench import TASK_REGISTRY

EPISODES = 3
MAX_STEPS = 1000

configs = {
    'cheetah-run': {'ppo': 'checkpoints/cheetah-run/ppo/model.zip', 'sac': None},
    'walker-walk': {'ppo': 'checkpoints/walker-walk/ppo/model.zip', 'sac': 'checkpoints/walker-walk/sac/model.zip'},
    'humanoid-stand': {'ppo': 'checkpoints/humanoid-stand/ppo/model.zip', 'sac': 'checkpoints/humanoid-stand/sac/model.zip'},
}

# Random returns (1000 steps, measured)
random_returns = {
    'cheetah-run': 3.40,
    'walker-walk': 30.24,
    'humanoid-stand': 8.55,
}

# Industry SOTA normalized scores (1M steps)
sota_scores = {
    'cheetah-run': {'tdmpc2': 878, 'dreamer': 887},
    'walker-walk': {'tdmpc2': 980, 'dreamer': 956},
    'humanoid-stand': {'tdmpc2': 873, 'dreamer': 945},
}

results = {}

for task, cfg in configs.items():
    domain, task_name = task.split('-', 1)
    env = suite.load(domain, task_name)
    goal = TASK_REGISTRY[task](env.physics, 0.05)
    kappa_thresh = goal.delta_K
    
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}")
    
    # PPO baseline
    ppo_ckpt = cfg['ppo']
    if ppo_ckpt and os.path.isfile(ppo_ckpt):
        try:
            ppo_adapter = SB3PPOAdapter(task_name=task, checkpoint_dir='checkpoints', auto_train_steps=0, verbose=0)
            if ppo_adapter.model is None:
                ppo_adapter.model = PPO.load(ppo_ckpt, env=ppo_adapter.gym_env)
                ppo_adapter._trained = True
            
            ppo_returns = []
            for ep in range(EPISODES):
                timestep = env.reset()
                ep_return = 0.0
                for step in range(MAX_STEPS):
                    action = ppo_adapter.choose_action(timestep)
                    timestep = env.step(action)
                    ep_return += float(timestep.reward or 0.0)
                    if timestep.last():
                        break
                ppo_returns.append(ep_return)
                print(f"  PPO ep{ep+1}: return={ep_return:.4f}")
            
            avg_ppo = np.mean(ppo_returns)
            print(f"  PPO avg: {avg_ppo:.4f}")
            results[f'{task}_PPO'] = avg_ppo
        except Exception as e:
            print(f"  PPO ERROR: {e}")
    
    # SAC baseline
    sac_ckpt = cfg['sac']
    if sac_ckpt and os.path.isfile(sac_ckpt):
        try:
            sac_adapter = SB3SACAdapter(task_name=task, checkpoint_dir='checkpoints', auto_train_steps=0, verbose=0)
            if sac_adapter.model is None:
                sac_adapter.model = SAC.load(sac_ckpt, env=sac_adapter.gym_env)
                sac_adapter._trained = True
            
            sac_returns = []
            for ep in range(EPISODES):
                timestep = env.reset()
                ep_return = 0.0
                for step in range(MAX_STEPS):
                    action = sac_adapter.choose_action(timestep)
                    timestep = env.step(action)
                    ep_return += float(timestep.reward or 0.0)
                    if timestep.last():
                        break
                sac_returns.append(ep_return)
                print(f"  SAC ep{ep+1}: return={ep_return:.4f}")
            
            avg_sac = np.mean(sac_returns)
            print(f"  SAC avg: {avg_sac:.4f}")
            results[f'{task}_SAC'] = avg_sac
        except Exception as e:
            print(f"  SAC ERROR: {e}")
    
    # Hybrid-PPO
    if f'{task}_PPO' in results:
        try:
            ppo_for_hybrid = SB3PPOAdapter(task_name=task, checkpoint_dir='checkpoints', auto_train_steps=0, verbose=0)
            if ppo_for_hybrid.model is None:
                ppo_for_hybrid.model = PPO.load(ppo_ckpt, env=ppo_for_hybrid.gym_env)
                ppo_for_hybrid._trained = True
            
            tc = get_controller_for_task(task, env.physics)
            hybrid = HybridSB3IDOAgent(sb3_adapter=ppo_for_hybrid, goal_eml=goal, task_name=task,
                                        kappa_thresh=kappa_thresh, task_controller=tc)
            
            hybrid_returns = []
            for ep in range(EPISODES):
                timestep = env.reset()
                hybrid.reset()
                ep_return = 0.0
                for step in range(MAX_STEPS):
                    action = hybrid.choose_action(timestep, physics=env.physics)
                    timestep = env.step(action)
                    ep_return += float(timestep.reward or 0.0)
                    if timestep.last():
                        break
                hybrid_returns.append(ep_return)
                print(f"  Hybrid-PPO ep{ep+1}: return={ep_return:.4f}")
            
            avg_hybrid = np.mean(hybrid_returns)
            print(f"  Hybrid-PPO avg: {avg_hybrid:.4f}")
            ratio = avg_hybrid / results[f'{task}_PPO'] if results[f'{task}_PPO'] > 0 else 0
            print(f"  Hybrid/PPO ratio: {ratio:.2f}x")
            results[f'{task}_HybridPPO'] = avg_hybrid
        except Exception as e:
            print(f"  Hybrid-PPO ERROR: {e}")
    
    # Hybrid-SAC
    if f'{task}_SAC' in results:
        try:
            sac_for_hybrid = SB3SACAdapter(task_name=task, checkpoint_dir='checkpoints', auto_train_steps=0, verbose=0)
            if sac_for_hybrid.model is None:
                sac_for_hybrid.model = SAC.load(sac_ckpt, env=sac_for_hybrid.gym_env)
                sac_for_hybrid._trained = True
            
            tc = get_controller_for_task(task, env.physics)
            hybrid_sac = HybridSB3IDOAgent(sb3_adapter=sac_for_hybrid, goal_eml=goal, task_name=task,
                                            kappa_thresh=kappa_thresh, task_controller=tc)
            
            hybrid_sac_returns = []
            for ep in range(EPISODES):
                timestep = env.reset()
                hybrid_sac.reset()
                ep_return = 0.0
                for step in range(MAX_STEPS):
                    action = hybrid_sac.choose_action(timestep, physics=env.physics)
                    timestep = env.step(action)
                    ep_return += float(timestep.reward or 0.0)
                    if timestep.last():
                        break
                hybrid_sac_returns.append(ep_return)
                print(f"  Hybrid-SAC ep{ep+1}: return={ep_return:.4f}")
            
            avg_hybrid_sac = np.mean(hybrid_sac_returns)
            print(f"  Hybrid-SAC avg: {avg_hybrid_sac:.4f}")
            ratio = avg_hybrid_sac / results[f'{task}_SAC'] if results[f'{task}_SAC'] > 0 else 0
            print(f"  Hybrid/SAC ratio: {ratio:.2f}x")
            results[f'{task}_HybridSAC'] = avg_hybrid_sac
        except Exception as e:
            print(f"  Hybrid-SAC ERROR: {e}")

# Summary with normalized scores
print(f"\n{'='*70}")
print(f"  GAP ANALYSIS: Our Scores vs Industry SOTA")
print(f"{'='*70}")
print(f"  Episode length: {MAX_STEPS} steps (standard dm_control)")
print(f"  Episodes: {EPISODES}")
print()

# Normalize: score = raw / max_possible * 1000 (approximate)
# dm_control rewards are in [0,1] per step, so max_return ≈ MAX_STEPS = 1000
max_possible = MAX_STEPS  # reward ∈ [0, ~1] per step → max total ≈ 1000

for task in configs.keys():
    ppo_raw = results.get(f'{task}_PPO', 0)
    sac_raw = results.get(f'{task}_SAC', 0)
    hybrid_ppo_raw = results.get(f'{task}_HybridPPO', 0)
    hybrid_sac_raw = results.get(f'{task}_HybridSAC', 0)
    
    random_ret = random_returns[task]
    
    ppo_norm = max(0, (ppo_raw - random_ret) / (max_possible - random_ret) * 1000)
    sac_norm = max(0, (sac_raw - random_ret) / (max_possible - random_ret) * 1000) if sac_raw > 0 else 0
    hybrid_ppo_norm = max(0, (hybrid_ppo_raw - random_ret) / (max_possible - random_ret) * 1000) if hybrid_ppo_raw > 0 else 0
    hybrid_sac_norm = max(0, (hybrid_sac_raw - random_ret) / (max_possible - random_ret) * 1000) if hybrid_sac_raw > 0 else 0
    
    tdmpc2 = sota_scores[task]['tdmpc2']
    dreamer = sota_scores[task]['dreamer']
    
    ppo_pct_of_sota = ppo_norm / tdmpc2 * 100
    
    print(f"  {task}:")
    print(f"    Our PPO:       raw={ppo_raw:.2f}, norm={ppo_norm:.1f}  ({ppo_pct_of_sota:.1f}% of SOTA)")
    if sac_raw > 0:
        sac_pct = sac_norm / tdmpc2 * 100
        print(f"    Our SAC:       raw={sac_raw:.2f}, norm={sac_norm:.1f}  ({sac_pct:.1f}% of SOTA)")
    if hybrid_ppo_raw > 0:
        hppo_pct = hybrid_ppo_norm / tdmpc2 * 100
        print(f"    Our Hybrid-PPO: raw={hybrid_ppo_raw:.2f}, norm={hybrid_ppo_norm:.1f}  ({hppo_pct:.1f}% of SOTA)")
    if hybrid_sac_raw > 0:
        hsac_pct = hybrid_sac_norm / tdmpc2 * 100
        print(f"    Our Hybrid-SAC: raw={hybrid_sac_raw:.2f}, norm={hybrid_sac_norm:.1f}  ({hsac_pct:.1f}% of SOTA)")
    print(f"    TD-MPC2 SOTA:  {tdmpc2}")
    print(f"    DreamerV3:     {dreamer}")
    print(f"    Gap to SOTA:   {tdmpc2 - max(ppo_norm, hybrid_ppo_norm):.1f} normalized points")
    print()
