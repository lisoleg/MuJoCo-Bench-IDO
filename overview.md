# v0.16.26 交付总结

## TL;DR
翻跟斗修复 + VLA指令修复 + 6个core模块接入benchmark循环 + Web UI更新 + 文档论文更新 + Git提交

## 交付概览
- **版本**: v0.16.26
- **服务器**: ✅ 运行中 (port 8080)
- **Git**: ✅ 已推送到 GitHub (commits 4851a55 + 2903a70)
- **新API**: /api/architecture ✅, /api/t_processor ✅

## Bug 修复

### Bug #1: 机器人翻跟斗（根因修复）
- **根因**: 地形场景下所有增益×0.7，包括pitch/roll稳定增益——在最需要稳定的地方反而降低了稳定性
- **修复**: 
  - 地形只缩放Z增益，角稳定增益保持满值
  - 降低检测阈值：12°/20°（原20°/30°）——更早干预
  - 高阻尼（4x）代替高力矩（3x）——防止过冲振荡
  - 主动角速度阻尼：|angvel|>2rad/s时提前加阻尼

### Bug #2: VLA指令无效果（根因修复）
- **根因**: VLA分支内调用_render_cam()，camera失败→异常→fallback到默认控制器；15%插值太慢不可见
- **修复**:
  - 解耦camera渲染与VLA预测（独立try/except）
  - 插值速率30%（原15%）——3倍速度，立即可见
  - 暂停时手动步进物理——即使viewer paused也能执行VLA动作

## 6个Core模块接入Benchmark循环

| 模块 | 接入点 | 每步输出 |
|------|--------|---------|
| KappaSnapTokenizer | η计算后 | token字符串 + 32维summary |
| T-Processor | action执行后 | η-ALU值 + ψ违规 |
| Three-Body | action执行 | sim-to-real gap |
| HG-PINN | 每步 | Hamiltonian能量统计 |
| Nine-Layer | API报告 | L0-L8层级状态 |
| ψ-LoRA | 离线收集 | DPO训练对 |

## Web UI更新
- 九层认知架构面板（L0-L8卡片，从/api/architecture动态加载）
- κ-Snap token实时显示
- T-Processor规格显示（65k gates, 3.3mW）
- 版本号更新至v0.16.26

## 文档论文更新
- README.md: 九层架构表、API列表、项目结构更新
- Paper Appendix C.22: T-Processor规格、KappaSnapTokenizer、Three-Body、HG-PINN、ψ-LoRA、S-Bridge LLM归因、benchmark接入表

## Git提交
- `4851a55`: v0.16.26 main commit (19 files, +4225 -172)
- `2903a70`: T-Processor API属性名修复
- 已推送到 https://github.com/lisoleg/MuJoCo-Bench-IDO.git

## 用户验证步骤
1. http://localhost:8080 — Dashboard（查看九层架构面板 + T-Processor规格）
2. http://localhost:8081 — 3D Viewer（地形场景验证翻跟斗修复）
3. http://localhost:8091 — ARM100（Submit Instruction验证机械臂运动）
4. `curl http://localhost:8080/api/architecture` — 九层架构API
5. `curl http://localhost:8080/api/t_processor` — T-Processor规格API
