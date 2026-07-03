"""IDO MuJoCo Agent package.

v0.6.0: Machine Conscience Audit Framework — includes PsiAnchor,
HybridSB3IDOAgent, AgentMode, SafeFuse for full audit trail.

v0.8.0 升级项 U5: 新增模块导出
  - SBridge (U5: MetaQuery 自我归因接口, 可选插件)
  - FuseLevel, FuseGradeResult, FuseOption (U1: SafeFuse 三级渐进)
"""
from agent.psi_anchor import PsiAnchor, PsiAnchorState
from agent.hybrid_sb3_ido_agent import HybridSB3IDOAgent, AgentMode
from agent.safe_fuse import SafeFuse, FuseLevel, FuseGradeResult, FuseOption
from agent.s_bridge import SBridge, SkillEntry
