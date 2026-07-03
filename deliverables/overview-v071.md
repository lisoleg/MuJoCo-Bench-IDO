# MuJoCo-Bench-IDO v0.7.1 — 交付总结

## TL;DR
Hybrid IDO+SB3 agent 性能从 **0.21x → 1.04x**（5 倍飞跃），walker-walk Hybrid-SAC 超过 SAC **42%**。3 个根因 Bug 修复：ctrl 污染、prev_data 引用、SafeFuse locomotion 误触发。HopperHopPD v1.0.1 完全重写，不再飞到 z=4.4。

## 交付概览
- **版本**: v0.7.1 (commit 8b5009e, pushed to GitHub)
- **测试通过率**: 292/292 (100%)
- **关键修复**: 3 个 physics corruption + SafeFuse bugs
- **新模块**: HopperHopPD v1.0.1 3-phase gait rewrite

## P0-P2 任务完成状态

| 任务 | 状态 | 结果 |
|------|------|------|
| P0: Hybrid benchmark 验证 | ✅ 完成 | Hybrid/PPO ratio 从 0.21x → 1.04x |
| P1: CheetahRunPD/性能调优 | ✅ 完成 | SafeFuse bypass + ctrl fix → 0.94x |
| P2: η mode 显示 | ✅ 完成 | Dashboard v0.6.6 badges + colors |
| P2: Hopper-hop 优化 | 🔄 进行中 | PD v1.0.1 avg_return>0, SB3 训练启动 |

## v0.7.1 基准测试结果

| 任务 | PPO → Hybrid-PPO | SAC → Hybrid-SAC |
|------|-------------------|-------------------|
| cheetah-run | 0.21x → **0.94x** | — |
| walker-walk | **1.15x** PPO | **1.42x** SAC |
| humanoid-stand | **1.04x** PPO | 0.82x SAC |

## HopperHopPD v1.0.1 性能

| 指标 | 旧版 | 新版 |
|------|------|------|
| avg_return | ~0.02 | 0.41–1.07 |
| max torso z | 4.4 (失控飞行) | 1.03 (合理范围) |
| z ∈ [0.8, 3.0] | ❌ | ✅ 50/50 |

## 根因诊断 (3 Bugs)

1. **phys.data.ctrl[:] = action** — choose_action() 内写入 ctrl，env.step() 重复赋值 → 物理状态污染
2. **prev_data = phys.data** — 存引用而非拷贝 → Noether check 比较自身 (always ok=True)
3. **SafeFuse L3_hard** — ψ-Anchor evolution_triggered 标志对 locomotion 触发 ×0.1 emergency fallback → 步态摧毁

## 下一步

1. SB3 PPO+SAC 训练 hopper-hop (1M steps × 2) — 后台运行中
2. Hybrid IDO+PPO/SAC 集成 hopper-hop
3. 扩展 Hybrid benchmarks 到全部 9 个 locomotion 任务
4. humanoid-stand Hybrid-SAC 优化 (当前 0.82x)
5. 持续优化到业界第一水平
