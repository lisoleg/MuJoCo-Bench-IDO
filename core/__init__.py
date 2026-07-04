"""IDO MuJoCo Core modules (κ-Snap, Noether, Goal-EML, PG-Gate, Schema, Logger, CQ).

v0.6.0: Machine Conscience Audit Framework — includes KappaSnapSchema,
KappaSnapLogger, MerkleChain, PGGate, BayesianIntent, ConscienceQuotient
for full audit trail.

v0.6.4: Triple-Entropy + Mao Rui Metric + Bian Saturation + Duodecimal Base
— new modules from 复合体理学 WeChat article series (6 articles).

v0.8.0 升级项 U3/U4/U6: 新增模块导出
  - PreAffect (U4: 内在信号 GRRR/PHEW/NEUTRAL)
  - KappaSnapJSONLWriter (U3: 步骤级 JSONL 审计输出)
  - EMLSemZipIC (U6: IC 计算 + Dead-Zero 过滤 + 毛睿重加权)
"""
from core.goal_eml_mj import GoalEML, make_humanoid_stand_eml, make_hopper_stand_eml, make_walker_run_eml, make_reacher_easy_eml
from core.kappa_snap_mj import gauss_ex_residual, FlowMatchingEtaPredictor, compute_merkle_snap_id
from core.noether_check_mj import noether_check_mj, NoetherViolation, _friction_cone_check
from core.kappa_snap_schema import KappaSnapSchema
from core.kappa_snap_logger import KappaSnapLogger, MerkleChain
from core.pg_gate import PGGate
from core.bayesian_intent import BayesianIntent
from core.cq import ConscienceQuotient
from core.triple_entropy import TripleEntropyLoss, EntropyConfig, ShannonEntropy, ThermodynamicEntropy, PsiAnchorGate
from core.mao_rui_metric import MaoRuiMetric, MaoRuiConfig, HyperEdge
from core.bian_saturation import BianSaturation, BianConfig
from core.duodecimal_base import DuodecimalBase, DuodecimalConfig
# ── v0.8.0 新增模块导出 ──
from core.pre_affect import PreAffect
from core.kappa_snap_jsonl import KappaSnapJSONLWriter, HermesTranslator
from core.eml_semzip_ic import EMLSemZipIC
# ── v0.16.19: GEL (Goal-EML Injection Loss) from VG-Pair theory ──
from core.gel_loss import GELLoss, GELConfig, compute_gel_from_step
