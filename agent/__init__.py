"""IDO MuJoCo Agent package.

v0.6.0: Machine Conscience Audit Framework — includes PsiAnchor,
HybridSB3IDOAgent, AgentMode, SafeFuse for full audit trail.

v0.8.0 升级项 U5: 新增模块导出
  - SBridge (U5: MetaQuery 自我归因接口, 可选插件)
  - FuseLevel, FuseGradeResult, FuseOption (U1: SafeFuse 三级渐进)

v0.17.0 升级: TOMAS Agent 全栈集成
  - TOMASMuJoCoWrapper (P->C->S 三层管线集成)
  - TOMASFailureAttributor (S-Layer LLM 驱动失败归因)
  - TOMASAgent (部署编排器 + MetaQuery 自我归因)
  - FootstepPlanner (足端轨迹规划 + ZMP 校验)
  - SupportPolygon (支撑多边形 + ZMP 包含检测)
  - Footstep, FootstepPlan (足步数据结构)
  - DeployStatus, DeployReport (部署报告)
  - MetaQueryType, MetaQueryResult (MetaQuery 自我归因)
  - SkillRecord (技能学习记录)
"""
from agent.psi_anchor import PsiAnchor, PsiAnchorState
from agent.hybrid_sb3_ido_agent import HybridSB3IDOAgent, AgentMode
from agent.safe_fuse import SafeFuse, FuseLevel, FuseGradeResult, FuseOption
from agent.s_bridge import SBridge, SkillEntry

# v0.17.0: TOMAS Agent 全栈集成模块
from agent.tomas_mujoco_wrapper import TOMASMuJoCoWrapper
from agent.failure_attribution import TOMASFailureAttributor, FailureAttributionResult
from agent.tomas_deploy import (
    TOMASAgent,
    DeployStatus,
    DeployReport,
    MetaQueryType,
    MetaQueryResult,
    SkillRecord,
)
from agent.footstep_planner import (
    FootstepPlanner,
    SupportPolygon,
    Footstep,
    FootstepPlan,
    FootSide,
    StepPhase,
)
