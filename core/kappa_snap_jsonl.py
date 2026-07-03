"""
κ-Snap JSONL 步骤级审计输出 + Hermes 翻译层
=============================================

v0.8.0 升级项 U3: κ-Snap 步骤级审计
  来源: 文12 κ-Snap specification + 文15 "缺细粒度因果链"

KappaSnapJSONLWriter: 步骤级 JSONL 文件输出
  - open(file_path): 打开 JSONL 文件
  - write_step(η, mode, fuse_level, pre_affect, ...): 写入一步审计记录
  - flush(): 刷新缓冲区到文件
  - close(): 关闭文件
  - query(snap_id): 查询特定步骤的审计记录

HermesTranslator: 私有标签→人可读映射
  - translate(private_label): 将系统内部私有标签翻译为人可读字符串
  - PRIVATE_MAP: 预定义映射表

JSONL 格式约定:
  - 每行一个 JSON 对象 (append-only)
  - 必含字段: snap_id, step, η, mode, fuse_level, pre_affect, noether_ok, timestamp
  - 可选字段: evidence_verified, safe_action, probe_type
  - 文件路径: logs/kappa_snap_{task_name}_{episode_id}.jsonl

默认不输出 — 需显式启用 (调用 open() 才开始写入).

Author: MuJoCo-Bench-IDO v0.8.0 — 升级项 U3
"""

import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List, Optional


# ── Hermes 翻译层: 私有标签 → 人可读映射 ──

PRIVATE_MAP: Dict[str, str] = {
    # ── v0.8.0 新增私有标签映射 ──
    "L3h": "ψ-Anchor 触发安全降级",
    "GRRR": "η停滞焦虑信号",
    "PHEW": "η突破释然信号",
    "EVC": "证据自校验完成",
    "FUSE_WARNING": "SafeFuse WARNING 级触发",
    "FUSE_INFO": "SafeFuse INFO 级触发(locomotion透明路由)",
    "FUSE_BLOCK": "SafeFuse BLOCK 级触发(严重违反)",
    "PRE_AFFECT_GRRR": "PreAffect GRRR — η停滞焦虑",
    "PRE_AFFECT_PHEW": "PreAffect PHEW — η突破释然",
    "PRE_AFFECT_NEUTRAL": "PreAffect NEUTRAL — 无特殊信号",
    # ── 已有标签映射 ──
    "L0": "系统级事件",
    "L1": "Noether守恒门违反",
    "L2": "ψ-Anchor知性限检查",
    "L3": "PG-Gate硬锚定夹",
    "L4": "自适应行为(Creative-Probe/漂移)",
    "L5": "任务级事件",
    "L6": "元管理层(ψ-Anchor进化)",
}


class HermesTranslator:
    """Hermes 翻译层 — 将系统内部私有标签翻译为人可读字符串.

    预定义 PRIVATE_MAP 映射表, 也支持用户扩展.

    Attributes:
        _map: 私有标签 → 人可读映射字典.
    """

    def __init__(self, extra_map: Optional[Dict[str, str]] = None) -> None:
        """初始化 Hermes 翻译层.

        Args:
            extra_map: 可选的额外映射字典, 合入预定义 PRIVATE_MAP.
        """
        self._map: Dict[str, str] = dict(PRIVATE_MAP)
        if extra_map is not None:
            self._map.update(extra_map)

    def translate(self, private_label: str) -> str:
        """将私有标签翻译为人可读字符串.

        Args:
            private_label: 系统内部私有标签字符串.

        Returns:
            人可读翻译字符串. 若标签不在映射表中, 返回原始标签.
        """
        return self._map.get(private_label, private_label)

    def add_mapping(self, private_label: str, human_readable: str) -> None:
        """添加新的映射条目.

        Args:
            private_label: 私有标签字符串.
            human_readable: 人可读翻译字符串.
        """
        self._map[private_label] = human_readable

    def get_map(self) -> Dict[str, str]:
        """返回完整映射表.

        Returns:
            私有标签 → 人可读映射字典.
        """
        return dict(self._map)


class KappaSnapJSONLWriter:
    """κ-Snap 步骤级 JSONL 审计文件输出.

    每步写入一条 JSONL 记录, 包含 η, mode, fuse_level, pre_affect 等字段.
    默认不输出 — 需显式调用 open() 才开始写入文件.

    Attributes:
        _file_path: JSONL 文件路径.
        _file: 已打开的文件对象 (None 表示未启用).
        _hermes: HermesTranslator 实例.
        _buffer: 内存中的步骤记录缓冲 (用于 query).
        _step_counter: 步骤计数器.
        _snap_id_counter: snap_id 递增计数器.
    """

    def __init__(self, hermes: Optional[HermesTranslator] = None) -> None:
        """初始化 KappaSnapJSONLWriter.

        Args:
            hermes: 可选的 HermesTranslator 实例. 若 None, 自动创建.
        """
        self._file_path: Optional[pathlib.Path] = None
        self._file: Optional[Any] = None
        self._hermes: HermesTranslator = hermes if hermes is not None else HermesTranslator()
        self._buffer: List[Dict[str, Any]] = []
        self._step_counter: int = 0
        self._snap_id_counter: int = 0
        self._prev_snap_id: str = "genesis"

    def open(self, file_path: str) -> None:
        """打开 JSONL 文件用于写入.

        创建文件目录 (如不存在), 打开文件追加模式.

        Args:
            file_path: JSONL 文件路径 (字符串或 pathlib.Path).
        """
        self._file_path = pathlib.Path(file_path)
        # 创建父目录
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        # 打开文件 (追加模式)
        self._file = open(self._file_path, 'a', encoding='utf-8')
        self._buffer = []
        self._step_counter = 0
        self._snap_id_counter = 0
        self._prev_snap_id = "genesis"

    def write_step(self,
                   eta: float,
                   mode: str,
                   fuse_level: str,
                   pre_affect: str,
                   noether_result: Optional[Dict[str, Any]] = None,
                   evidence_verified: Optional[bool] = None) -> str:
        """写入一步审计记录到 JSONL 文件.

        记录格式:
          {
            "snap_id": "...",
            "step": N,
            "η": eta_value,
            "mode": "EXPLOIT/EXPLORE/SAFE",
            "fuse_level": "NORMAL/WARNING/BLOCK/INFO",
            "pre_affect": "GRRR/PHEW/NEUTRAL",
            "noether_ok": bool,
            "timestamp": unix_ts,
            "evidence_verified": bool (可选),
            "hermes_translation": {...} (翻译后的私有标签)
          }

        Args:
            eta: κ-Snap 残差 η 值.
            mode: Agent 模式 (EXPLOIT/EXPLORE/SAFE).
            fuse_level: SafeFuse 级别 (NORMAL/WARNING/BLOCK/INFO 或传统级别).
            pre_affect: PreAffect 信号 (GRRR/PHEW/NEUTRAL).
            noether_result: Noether 检查结果 Dict.
            evidence_verified: 证据校验标记 (可选).

        Returns:
            本步的 snap_id 字符串.
        """
        # ── 计算 snap_id ──
        hash_input: str = self._prev_snap_id + str(eta) + str(mode) + str(self._step_counter)
        snap_hash: str = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
        snap_id: str = f"snap_{self._step_counter}_{snap_hash}"
        self._prev_snap_id = snap_id

        # ── 构建 JSONL 记录 ──
        noether_ok: bool = True
        if noether_result is not None:
            noether_ok = noether_result.get("ok", True)

        record: Dict[str, Any] = {
            "snap_id": snap_id,
            "step": self._step_counter,
            "η": eta,
            "mode": mode,
            "fuse_level": fuse_level,
            "pre_affect": pre_affect,
            "noether_ok": noether_ok,
            "timestamp": time.time(),
        }

        # 可选字段: evidence_verified
        if evidence_verified is not None:
            record["evidence_verified"] = evidence_verified

        # ── Hermes 翻译层处理 ──
        hermes_translations: Dict[str, str] = {}
        # 翻译 fuse_level
        hermes_translations["fuse_level"] = self._hermes.translate(fuse_level)
        # 翻译 pre_affect
        pre_affect_label = f"PRE_AFFECT_{pre_affect}" if pre_affect != "NEUTRAL" else "PRE_AFFECT_NEUTRAL"
        hermes_translations["pre_affect"] = self._hermes.translate(pre_affect_label)
        # 翻译 mode (直接映射)
        hermes_translations["mode"] = self._hermes.translate(mode)
        record["hermes_translation"] = hermes_translations

        # ── 写入内存缓冲 ──
        self._buffer.append(record)

        # ── 写入 JSONL 文件 ──
        if self._file is not None:
            line: str = json.dumps(record, ensure_ascii=False, default=str)
            self._file.write(line + "\n")
            # 不每次 flush — 由 flush() 方法控制

        self._step_counter += 1
        return snap_id

    def flush(self) -> None:
        """刷新缓冲区到文件 (将 pending 数据写入磁盘)."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """关闭 JSONL 文件 (flush + close)."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def query(self, snap_id: str) -> Dict[str, Any]:
        """查询特定 snap_id 的步骤审计记录.

        从内存缓冲中查找匹配 snap_id 的记录.

        Args:
            snap_id: 要查询的 snap_id 字符串.

        Returns:
            匹配的记录 Dict. 若未找到, 返回空 Dict.
        """
        for record in self._buffer:
            if record.get("snap_id") == snap_id:
                return dict(record)
        return {}

    def get_buffer(self) -> List[Dict[str, Any]]:
        """返回完整内存缓冲.

        Returns:
            所有步骤记录列表.
        """
        return list(self._buffer)

    def reset(self) -> None:
        """重置写入器状态 (关闭文件, 清空缓冲)."""
        self.close()
        self._buffer = []
        self._step_counter = 0
        self._snap_id_counter = 0
        self._prev_snap_id = "genesis"
