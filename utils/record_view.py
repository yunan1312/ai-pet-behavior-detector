"""记录展示相关的纯函数，供历史面板、详情页与导出共用（便于测试）。"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from utils.constants import BEHAVIOR_DISPLAY_NAMES

SOURCE_LABELS = {
    "camera": "摄像头",
    "video": "视频",
    "image": "图片",
}


def parse_timestamp(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def format_timestamp(value: str | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    dt = parse_timestamp(value)
    return dt.strftime(fmt) if dt else (value or "--")


def format_duration(seconds: Optional[float]) -> str:
    try:
        total = int(round(float(seconds or 0)))
    except (TypeError, ValueError):
        return "--"
    if total <= 0:
        return "--"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_offset(seconds: Optional[float]) -> str:
    """格式化行为在视频中的时间偏移，0 秒显示为 00:00。"""
    if seconds is None:
        return "--"
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "--"
    if total < 0:
        return "--"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def source_label(record: Dict) -> str:
    source_type = record.get("source_type", "unknown")
    kind = SOURCE_LABELS.get(source_type, source_type or "未知")
    path = record.get("source_path") or ""
    if path:
        name = os.path.basename(path)
        if name.lower() in {"camera 0", "camera 1", "camera 2"}:
            return kind
        return f"{kind} · {name}"
    return kind


def source_kind(record: Dict) -> str:
    return SOURCE_LABELS.get(record.get("source_type", ""), "其他")


def behavior_counts(record: Dict) -> List[Tuple[str, int]]:
    counts: Dict[str, int] = {}
    for item in record.get("behaviors", []) or []:
        behavior = item.get("behavior", "")
        if not behavior or behavior in {"idle", "error"}:
            continue
        counts[behavior] = counts.get(behavior, 0) + 1
    return sorted(counts.items(), key=lambda pair: pair[1], reverse=True)


def dominant_behavior(record: Dict) -> str:
    counts = behavior_counts(record)
    if counts:
        return BEHAVIOR_DISPLAY_NAMES.get(counts[0][0], counts[0][0])
    return "—"


def top_confidence(record: Dict) -> float:
    values = [
        float(item.get("confidence", 0.0) or 0.0)
        for item in record.get("behaviors", []) or []
        if item.get("behavior") not in {None, "", "idle", "error"}
    ]
    return max(values) if values else 0.0


def record_summary(record: Dict, limit: int = 60) -> str:
    summary = (record.get("summary") or "").strip()
    if not summary:
        behaviors = record.get("behaviors", []) or []
        if behaviors:
            summary = behaviors[0].get("intent", "") or ""
    if len(summary) > limit:
        summary = summary[: limit - 1] + "…"
    return summary or "—"


def detection_count(record: Dict) -> int:
    return sum(
        1
        for item in record.get("behaviors", []) or []
        if item.get("behavior") not in {None, "", "idle", "error"}
    )


def behavior_distribution(record: Dict) -> List[Dict]:
    total = max(1, detection_count(record))
    result = []
    for behavior, count in behavior_counts(record):
        result.append(
            {
                "behavior": behavior,
                "display": BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior),
                "count": count,
                "percent": count / total,
            }
        )
    return result
