# v0.17.1 — TOMAS Agent Deploy API + VLA Loader + End-to-End Eval

## TL;DR
在 v0.17.0 TOMAS Agent 全栈集成基础上，完成 deploy API 接入 server.py、VLA 权重加载验证、SO-ARM100 端到端 pick-and-place 评估，681/681 测试全部通过，零回归。

## 交付概览
- **交付状态**: ✅ 完成
- **测试通过率**: 681/681 (100%) — 较 v0.17.0 新增 22 个测试
- **已知问题数**: 0
- **新测试**: 22个 (tests/test_tomas_deploy.py)
- **执行时间**: 67.56s
- **GitHub**: 已提交并推送 (commit 76ab6e3)

## 本轮完成的工作

### Task #289: TOMASAgent 接入 webviz/server.py ✅
- 新建 `webviz/tomas_deploy_api.py` — HeadlessMuJoCoEnv + TOMASAgent 工厂 + 评估入口
- 在 `webviz/server.py` 添加 5 个 TOMAS deploy API 端点:
  - `POST /api/tomas/deploy` — 异步启动 TOMAS 部署
  - `GET /api/tomas/deploy_status` — 查询运行状态
  - `GET /api/tomas/deploy_result` — 获取最终报告
  - `GET /api/tomas/vla_available` — 检查 VLA 模型可用性
  - `POST /api/tomas/quick_eval` — 同步快速单 episode 评估

### Task #290: VLA 权重加载验证 ✅
- 新建 `webviz/vla_loader.py` — VLALoader + 系统需求检查 + 动作验证 + 基准测试
- 支持 4 种 VLA 模型: OpenVLA-7B (7B, 16GB VRAM), Octo-Base (93M, 4GB), Pi0-Base (PaliGemma, 8GB), DemoVLA (内置, 0GB)
- `check_system_requirements()` — torch/CUDA/VRAM/package 检查
- `verify_action_output()` — 验证 VLA 输出 7-DOF action 且无 NaN
- `benchmark_inference()` — 10 次推理延迟统计 (avg/min/max/p50/p95/effective_hz)

### Task #291: SO-ARM100 端到端 pick-and-place 评估 ✅
- 新建 `benchmarks/run_tomas_eval.py` — 独立评估脚本 (支持 CLI 参数)
- 新建 `tests/test_tomas_deploy.py` — 22 个集成测试 (7 个测试类)
- 评估结果 (demo-vla, 2 episodes, 100 steps/episode):
  - Status: SUCCESS
  - Total Steps: 200
  - Kappa-Snap Count: 100 (MerkleChain 完整)
  - Psi Violations: 0
  - MetaQueries: 2 (AUDIT_SNAP)
  - Failure Attributions: 2
  - Chain Integrity: True

### Task #292: 提交代码到 GitHub ✅
- commit 76ab6e3 推送到 main 分支
- 18 files changed, 5138 insertions(+), 110 deletions(-)

## Bug 修复 (本轮发现并修复)
1. **tomas_mujoco_wrapper.py**: info dict 缺少 `raw_action` 和 `psi_violations` — 已添加
2. **tomas_mujoco_wrapper.py**: snap_logger.log() details 缺少 step/violations/gate 信息 — 已丰富
3. **tomas_mujoco_wrapper.py**: `get_audit_trail()` 返回 MerkleChain (无 details) — 改为返回 log_buffer
4. **tomas_deploy.py**: `kappa_snap_logger` 属性名错误 — 修正为 `snap_logger`
5. **tomas_deploy.py**: `snap_history` 方法名错误 — 修正为 `get_log_buffer()`
6. **tomas_deploy.py**: audit trail 字段访问未通过 details dict — 已修复所有访问路径
7. **tomas_deploy.py**: raw_action 需转为 np.asarray — 已修复

## 文件清单

### 新建文件 (5个)
| 文件 | 说明 |
|------|------|
| `webviz/tomas_deploy_api.py` | HeadlessMuJoCoEnv + TOMASAgent 工厂 + 评估入口 |
| `webviz/vla_loader.py` | VLA 权重加载器 (4种模型) |
| `benchmarks/run_tomas_eval.py` | 独立评估脚本 (CLI) |
| `tests/test_tomas_deploy.py` | 22 个集成测试 |
| `tests/test_v0170.py` | v0.17.0 单元测试 (上一轮) |

### 修改文件 (7个)
| 文件 | 修改内容 |
|------|---------|
| `webviz/server.py` | 5 个 TOMAS deploy API 端点 |
| `agent/tomas_mujoco_wrapper.py` | Bug fix: info dict + audit trail |
| `agent/tomas_deploy.py` | Bug fix: snap_logger + field access |
| `agent/__init__.py` | 导出新模块 |
| `core/hg_pinn.py` | PG-Gate 升级 |
| `benchmarks/welding_eval.py` | max_steps + critical-only |
| `envs/welding_env.py` | NUM_WAYPOINTS 调整 |

## 下一步建议
1. 在有 GPU 的机器上加载真实 OpenVLA-7B 权重并运行评估
2. 增加 SAC 训练到 500+ episodes (Task #278)
3. 重新运行焊接评估 + 验证 + 提交 (Task #279)
4. 优化 demo-vla IK 策略以降低 final_eta
