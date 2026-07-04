"""
WeldingEMLDistillation — 八元数EML因果蒸馏网络
==============================================

基于八元数非结合代数的EML因果蒸馏网络。
将高维物理读数映射到八元数空间，通过Φ算子计算η残差。

PyTorch可用时使用GPU加速，否则numpy回退。

损失: L = L_eta(BCE) + L_p(MSE on q0) + L_norm(单位长度约束)

参考: 章锋论文附录I

Author: MuJoCo-Bench-IDO v0.3.0
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.octonion_ops import OctonionOps, OctonionEMLNode

__all__ = [
    "WeldingEMLDistiller",
    "DistillationLoss",
    "generate_eml_candidates_from_stats",
    "HAS_TORCH",
]

# κ-Phase: 可选依赖检测
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None  # type: ignore


if HAS_TORCH:

    class _OctonionOpsTorch:
        """PyTorch八元数运算（批量）."""

        @staticmethod
        def mul(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
            """批量八元数乘法, (N,8)×(N,8)→(N,8)."""
            a0,a1,a2,a3,a4,a5,a6,a7 = [a[:,i] for i in range(8)]
            b0,b1,b2,b3,b4,b5,b6,b7 = [b[:,i] for i in range(8)]
            res = torch.stack([
                a0*b0-a1*b1-a2*b2-a3*b3-a4*b4-a5*b5-a6*b6-a7*b7,
                a0*b1+a1*b0+a2*b3-a3*b2+a4*b5-a5*b4-a6*b7+a7*b6,
                a0*b2-a1*b3+a2*b0+a3*b1+a4*b6+a5*b7-a6*b4-a7*b5,
                a0*b3+a1*b2-a2*b1+a3*b0+a4*b7-a5*b6+a6*b5-a7*b4,
                a0*b4-a1*b5-a2*b6-a3*b7+a4*b0+a5*b1+a6*b2+a7*b3,
                a0*b5+a1*b4-a2*b7+a3*b6-a4*b1+a5*b0-a6*b3+a7*b2,
                a0*b6+a1*b7+a2*b4-a3*b5-a4*b2+a5*b3+a6*b0-a7*b1,
                a0*b7-a1*b6+a2*b5+a3*b4-a4*b3-a5*b2+a6*b1+a7*b0,
            ], dim=1)
            return res

        @staticmethod
        def phi(q: "torch.Tensor", omega: "torch.Tensor") -> "torch.Tensor":
            """Φ(q,ω) = (q·ω)·q 左结合."""
            q_omega = _OctonionOpsTorch.mul(q, omega)
            return _OctonionOpsTorch.mul(q_omega, q)


    class WeldingEMLDistiller(nn.Module):
        """焊接EML八元数蒸馏网络 (PyTorch).

        输入: 归一化物理读数(8维)
        输出: 八元数q(8维) + 目标陪集omega(8维) + Φ(q,omega)(8维)
        """

        def __init__(self, hidden_dim: int = 128) -> None:
            super().__init__()
            self.feat = nn.Sequential(
                nn.Linear(8, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            )
            self.to_oct = nn.Linear(hidden_dim, 8)
            self.omega_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                nn.Linear(hidden_dim, 8),
            )

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            h = self.feat(x)
            q = self.to_oct(h)
            omega = self.omega_net(h)
            phi_result = _OctonionOpsTorch.phi(q, omega)
            return q, omega, phi_result


    class DistillationLoss(nn.Module):
        """蒸馏损失: L = L_eta + L_p + L_norm.

        L_eta: BCE(sum(p²), y_eta) — η残差
        L_p:   0.5 × MSE(q0, y_d) — 物理量回归
        L_norm: 0.01 × mean((sum(q²)-1)²) — 单位长度约束
        """

        def __init__(self, le: float = 1.0, lp: float = 0.5, ln: float = 0.01) -> None:
            super().__init__()
            self.le = le
            self.lp = lp
            self.ln = ln
            self.bce = nn.BCELoss()
            self.mse = nn.MSELoss()

        def forward(
            self,
            q: "torch.Tensor",
            omega: "torch.Tensor",
            p: "torch.Tensor",
            y_eta: "torch.Tensor",
            y_d: "torch.Tensor",
        ) -> "torch.Tensor":
            pn = (torch.sum(q * q, 1, keepdim=True) - 1.0) ** 2
            eta_pred = torch.clamp(torch.sum(p * p, 1, keepdim=True), 0, 1)
            return (
                self.le * self.bce(eta_pred, y_eta)
                + self.lp * self.mse(q[:, 0:1], y_d)
                + self.ln * torch.mean(pn)
            )

else:
    # numpy回退
    class WeldingEMLDistiller:  # type: ignore
        """焊接EML蒸馏网络 (numpy回退版).

        简化MLP实现，无梯度反向传播。
        仅支持前向推理。
        """

        def __init__(self, hidden_dim: int = 128) -> None:
            self.hidden_dim = hidden_dim
            rng = np.random.default_rng(42)
            self.W1 = rng.standard_normal((8, hidden_dim)) * 0.1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * 0.1
            self.b2 = np.zeros(hidden_dim)
            self.W_q = rng.standard_normal((hidden_dim, 8)) * 0.1
            self.b_q = np.zeros(8)
            self.W_om = rng.standard_normal((8, hidden_dim)) * 0.1
            self.b_om = np.zeros(hidden_dim)
            self.W_om2 = rng.standard_normal((hidden_dim, 8)) * 0.1
            self.b_om2 = np.zeros(8)

        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            h1 = np.maximum(0, x @ self.W1 + self.b1)
            h2 = np.maximum(0, h1 @ self.W2 + self.b2)
            q = h2 @ self.W_q + self.b_q
            h_om = np.maximum(0, x @ self.W_om + self.b_om)
            omega = h_om @ self.W_om2 + self.b_om2
            phi_result = np.array([OctonionOps.phi(q[i], omega[i]) for i in range(len(q))])
            return q, omega, phi_result

        def __call__(self, x):
            return self.forward(x)


    class DistillationLoss:  # type: ignore
        """蒸馏损失 (numpy回退版)."""

        def __init__(self, le: float = 1.0, lp: float = 0.5, ln: float = 0.01) -> None:
            self.le = le
            self.lp = lp
            self.ln = ln

        def __call__(
            self,
            q: np.ndarray,
            omega: np.ndarray,
            p: np.ndarray,
            y_eta: np.ndarray,
            y_d: np.ndarray,
        ) -> float:
            eta_pred = np.clip(np.sum(p ** 2, axis=1, keepdims=True), 0, 1)
            bce = -np.mean(y_eta * np.log(eta_pred + 1e-8) + (1-y_eta) * np.log(1-eta_pred + 1e-8))
            mse = np.mean((q[:, 0:1] - y_d) ** 2)
            pn = np.mean((np.sum(q ** 2, axis=1, keepdims=True) - 1) ** 2)
            return self.le * bce + self.lp * mse + self.ln * pn


def generate_eml_candidates_from_stats(all_stats: List[dict]) -> List[dict]:
    """从DreamerV3训练统计生成EML候选节点.

    取η最小的前10% episodes，转换为八元数EML节点。

    Args:
        all_stats: DreamerV3训练统计列表，每个元素包含:
            - final_I: 最终电流
            - final_V: 最终电压
            - final_eta: 最终η残差
            - final_porosity: 最终气孔率
            - final_distortion: 最终变形量
            - stick_out: 干伸长
            - steps: 步数
            - episode: episode编号
            - reward: 累计奖励

    Returns:
        EML候选节点列表.
    """
    if not all_stats:
        return []

    sorted_by_eta = sorted(all_stats, key=lambda x: x.get("final_eta", 0))
    top_n = max(1, len(sorted_by_eta) // 10)
    candidates = sorted_by_eta[:top_n]

    eml_nodes: List[dict] = []
    for c in candidates:
        node = {
            "node_id": f"SIM_{int(c.get('episode', 0))}",
            "q_state": [
                c.get("final_I", 120) / 250.0,
                c.get("final_V", 18) / 40.0,
                c.get("final_eta", 0),
                c.get("final_porosity", 0),
                c.get("final_distortion", 0),
                c.get("stick_out", 8) / 15.0,
                c.get("steps", 0) / 500.0,
                0.0,
            ],
            "psi_constraints": {
                "max_current": 200.0,
                "min_stick_out": 3.0,
                "max_stick_out": 15.0,
                "porosity_threshold": 0.3,
            },
            "meta": {
                "source": "DreamerV3_MuJoCo",
                "episode": c.get("episode", 0),
                "reward": c.get("reward", 0),
                "material": "SUS304",
                "thickness_mm": 2.0,
                "joint_type": "Fillet",
            },
        }
        eml_nodes.append(node)

    return eml_nodes


def _self_test() -> bool:
    """自测."""
    print(f"[welding_eml_distillation] HAS_TORCH={HAS_TORCH}")

    # 测试网络前向传播
    distiller = WeldingEMLDistiller(hidden_dim=64)

    if HAS_TORCH:
        x = torch.randn(4, 8)
        q, omega, phi_result = distiller(x)
        assert q.shape == (4, 8), f"q shape should be (4,8), got {q.shape}"
        assert omega.shape == (4, 8)
        assert phi_result.shape == (4, 8)
        print(f"  PyTorch forward: q={q.shape}, omega={omega.shape}, phi={phi_result.shape}")

        # 测试损失计算
        loss_fn = DistillationLoss()
        y_eta = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        y_d = torch.tensor([[0.5], [-0.3], [0.8], [-0.1]])
        loss = loss_fn(q, omega, phi_result, y_eta, y_d)
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        print(f"  Loss: {loss.item():.4f}")
    else:
        x = np.random.randn(4, 8)
        q, omega, phi_result = distiller(x)
        assert q.shape == (4, 8)
        loss_fn = DistillationLoss()
        y_eta = np.array([[1.0], [0.0], [1.0], [0.0]])
        y_d = np.array([[0.5], [-0.3], [0.8], [-0.1]])
        loss = loss_fn(q, omega, phi_result, y_eta, y_d)
        assert loss > 0
        print(f"  Numpy forward: q={q.shape}, loss={loss:.4f}")

    # 测试EML候选生成
    stats = [
        {"episode": i, "final_I": 120+i, "final_V": 18, "final_eta": 0.1-i*0.01,
         "final_porosity": 0.05, "final_distortion": 0.01, "stick_out": 8,
         "steps": 200, "reward": -10+i}
        for i in range(20)
    ]
    candidates = generate_eml_candidates_from_stats(stats)
    assert len(candidates) == 2, f"Should get top 10% = 2 candidates, got {len(candidates)}"
    assert candidates[0]["q_state"][2] <= candidates[1]["q_state"][2], "Should be sorted by eta"
    print(f"  EML candidates: {len(candidates)} from {len(stats)} stats")

    print("[welding_eml_distillation] Self-test PASSED.")
    return True


if __name__ == "__main__":
    _self_test()
