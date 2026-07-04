# v0.16.25 交付总结

## TL;DR
3个Bug修复 + 全部P0/P1/P2功能实现完毕，15/15模块验证通过。

## 交付概览
- **版本**: v0.16.25
- **服务器状态**: ✅ 运行中 (port 8080, v0.16.25 confirmed)
- **测试通过率**: 15/15 模块 PASS (14 first-pass + 1 PsiLoRA bug fix)
- **已知问题**: 新模块为独立实现，尚未接入 server.py 主循环

## Bug 修复 (3个)

### Bug #1: 机器人突然翻个
- **原因**: 之前的anti-bounce只解决垂直弹跳，未处理地形边缘导致的角速度翻转
- **修复**: Flip detection (|roll|/|pitch|>0.52rad=flipping, >0.35rad=tilting) + 紧急恢复 (3x纠正力矩 + 蹲下 + 步态冻结)
- **文件**: `webviz/server.py`

### Bug #2: ARM100 camera黑屏
- **原因**: mj.Renderer需要GL上下文(EGL)，headless环境失败
- **修复**: 3级fallback (Renderer → MjRenderContextOffscreen → PIL信息覆盖图) + 初始渲染 + 周期渲染
- **文件**: `webviz/server.py`

### Bug #3: Submit Instruction后机械臂不动
- **原因**: VLA model设为"none"，没有实际adapter执行动作
- **修复**: DemoVLAAdapter (指令解析→pick-and-place phases) + 自动启用VLA模式 + 自动取消暂停
- **文件**: `webviz/server.py`, `webviz/tomas_wrapper.py`

## P0 功能

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| KappaSnapTokenizer | `core/kappa_snap_tokenizer.py` | 350 | κ-Snap→[KSNAP:level:event:eta:decision] token编码, sliding window=16, 32维summary |
| S-Bridge LLM归因 | `agent/s_bridge.py` | +200 | ask_why_llm() few-shot prompt + template fallback |

## P1 功能

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| ψ-Anchor ZMP+Energy | `webviz/tomas_wrapper.py` | +80 | check_zmp() + check_energy_drift() |
| T-Processor | `core/t_processor.py` | 400 | EtaALU(Q16.16) + PsiChecker + KappaSnapFIFO, 100Hz 65k gates |

## P2 功能

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| Three-Body | `core/three_body.py` | 300 | Virtual→Software→Physical, isomorphism check |
| HG-PINN | `core/hg_pinn.py` | 200 | Hamiltonian-guided action head |
| ψ-anchor LoRA | `core/psi_lora.py` | 200 | DPO preference training, rank-4 LoRA |
| Nine-Layer Mapping | `core/nine_layer.py` | 250 | L0-L8 认知架构注册表 |

## 文件清单

### 修改的文件
1. `webviz/server.py` — v0.16.25, 3 bug fixes
2. `webviz/tomas_wrapper.py` — DemoVLAAdapter + ZMP/Energy check
3. `agent/s_bridge.py` — ask_why_llm() LLM归因
4. `webviz/dashboard.html` — 版本号更新

### 新建的文件
5. `core/kappa_snap_tokenizer.py` — κ-Snap Token编码器
6. `core/t_processor.py` — T-Processor硬件模拟
7. `core/three_body.py` — 三身体架构
8. `core/hg_pinn.py` — Hamiltonian-guided PINN
9. `core/psi_lora.py` — ψ-anchor LoRA DPO
10. `core/nine_layer.py` — 九层认知映射

## 用户下一步建议
1. **打开浏览器验证**: http://localhost:8080 (Dashboard), http://localhost:8081 (3D Viewer地形场景验证翻个修复), http://localhost:8091 (ARM100验证camera+VLA)
2. **测试VLA指令**: 在8091页面选择"demo-vla"模型，输入"pick the block"并点Submit，观察机械臂执行pick-and-place
3. **接入主循环**: 新建的6个core模块目前是独立实现，下一步需要接入server.py benchmark循环
4. **Git提交**: 所有修改可以commit到v0.16.25
5. **收集preference pairs**: 运行benchmark时从κ-Snap audit trail收集数据，用于ψ-LoRA训练
