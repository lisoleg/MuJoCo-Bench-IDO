# v0.17.2 — eta 计算修复 + 焊接 Viewer 端口生命周期修复 + SAC 训练

## TL;DR
修复 TOMASMuJoCoWrapper 中 eta 计算的核心 Bug（用关节角度而非物理距离），avg_eta 从 1.463 降到 0.103（-93%）。修复焊接 3D Viewer ViserServer 端口泄漏问题（"running: undefined" 和 timeout 错误），实现完整的端口生命周期管理。681/681 测试全部通过，零回归。同时完成 SAC 焊接训练 500 episodes。

## 交付概览
- **交付状态**: ✅ eta 修复 + 焊接 Viewer 端口修复完成；✅ SAC 训练 checkpoint 已保存
- **测试通过率**: 681/681 (100%)
- **已知问题数**: 0
- **GitHub**: 已提交并推送

## 本轮完成的工作

### eta 计算修复 (v0.17.2) ✅

**Bug 描述**:
- `TOMASMuJoCoWrapper._compute_eta()` 使用 `obs[:3]` (关节角度 [0.0, 0.3, -0.5]) 与 `goal=[0,0,0]` 计算 L2 距离
- 得到 ||关节角度|| ≈ 0.58-1.5，而非真实物理距离 ||gripper_pos - target_pos|| ≈ 0.12-0.41
- 导致 TOMAS 评估报告 avg_eta=1.463, final_eta=1.490（远高于实际物理距离）

**修复方案** (`agent/tomas_mujoco_wrapper.py`):
1. `_compute_eta()`: 优先使用 `obs[14:17]` (HeadlessMuJoCoEnv 提供的 gripper-to-target 距离向量)
2. `step()`: 优先使用 env 返回的 `info["eta"]`（已由 HeadlessMuJoCoEnv 计算为 `||gripper_pos - target_pos||`）

**验证结果**:
```
Status:           success
Total Steps:      200
Avg Eta:          0.103415  (was 1.463046, -93%)
Final Eta:        0.119896  (was 1.490389, -92%)
Psi Violations:   0
Kappa-Snap Count: 100
Chain Integrity:  True
```

### 焊接 3D Viewer 端口生命周期修复 ✅

**问题**: 用户点击 "Start Welding" 后显示 "running: undefined" 或 "[Timeout] Welding viewer startup timed out"

**根因**: `welding_stop()` 未调用 `persistent_server.stop()`，导致 8097-8102 端口被僵尸 ViserServer 占用。新启动时 6 次端口绑定全失败 → 返回 timeout → 前端缺少 timeout 分支显示 "Running: undefined"

**修复方案** (4 处后端修改 + 1 处前端修改):

1. **`_launch_welding_viewer()` 端口管理** (`webviz/server.py`):
   - 端口范围从 6 个 (8097-8102) 扩展到 16 个 (8097-8112)
   - 复用 persistent server 前用 `socket.connect_ex` 验证端口确实在监听
   - Stale server 先 `.stop()` + `sleep(1.0)` 等 OS 释放端口

2. **Viewer 异常退出时清理** (`webviz/server.py`):
   - except 块中调用 `persistent_server.stop()` 释放端口
   - 重置 `welding_persistent_server = None`

3. **`welding_stop()` 彻底清理** (`webviz/server.py`):
   - 调用 `persistent_server.stop()` 停止 ViserServer
   - `sleep(1.5)` 等待 OS 释放 socket

4. **`welding_start()` timeout 响应** (`webviz/server.py`):
   - 添加 `weld_type` 字段到 timeout 响应

5. **前端 timeout 状态处理** (`webviz/dashboard.html`):
   - 添加 `data.status === 'timeout'` 分支
   - 显示橙色 "[Timeout]" 提示
   - 自动调用 `/api/welding/stop` 清理后提示用户重试

**验证**: 连续 3 次 start→stop 循环全部成功，端口每次正确释放

### SAC 焊接训练 500 Episodes ✅
- 命令: `sac_weld_train.py --episodes 500 --steps 1000 --weld-type flat`
- 使用 Stable-Baselines3 SAC 2.9.0 + WeldingGymWrapper + KSnapCallback
- Checkpoint: `checkpoints/sac_weld/sac_weld_flat.zip` (1.47MB)
- eta 持续下降 (0.31→0.08)，reward 改善中

## 文件清单

### 修改文件 (3个)
| 文件 | 修改内容 |
|------|---------|
| `agent/tomas_mujoco_wrapper.py` | `_compute_eta()` + `step()` eta 计算修复 |
| `webviz/server.py` | 焊接 Viewer 端口生命周期管理 (4 处修改) |
| `webviz/dashboard.html` | 前端 timeout 状态处理分支 |

### 产出文件 (1个)
| 文件 | 说明 |
|------|------|
| `benchmarks/tomas_eval_report.json` | 修复后评估报告 |

### Checkpoint (1个)
| 文件 | 说明 |
|------|------|
| `checkpoints/sac_weld/sac_weld_flat.zip` | SAC 焊接训练 checkpoint (1.47MB) |

## 下一步建议
1. SAC 训练完成后运行焊接评估: `python benchmarks/welding_eval.py`
2. 在有 GPU 的机器上加载真实 OpenVLA-7B 权重并运行评估
3. 考虑增加 SAC 训练的 parallel environment 数量以加速训练
4. 扩展焊接评估到更多焊缝类型 (horizontal/vertical/overhead)
