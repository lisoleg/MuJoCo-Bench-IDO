"""IDO MuJoCo Core modules (κ-Snap, Noether, Goal-EML, PG-Gate, Schema, Logger, CQ).

v0.6.0: Machine Conscience Audit Framework — includes KappaSnapSchema,
KappaSnapLogger, MerkleChain, PGGate, BayesianIntent, ConscienceQuotient
for full audit trail.

v0.6.4: Triple-Entropy + Mao Rui Metric + Bian Saturation + Duodecimal Base
— new modules from 复合体理学 WeChat article series (6 articles).
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
