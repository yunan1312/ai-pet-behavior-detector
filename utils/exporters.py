"""历史记录导出：CSV 汇总与 Markdown 单条报告。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from utils.constants import BEHAVIOR_DISPLAY_NAMES
from utils.record_view import (
    behavior_distribution,
    detection_count,
    dominant_behavior,
    format_duration,
    format_offset,
    format_timestamp,
    record_summary,
    source_kind,
    source_label,
    top_confidence,
)

CSV_HEADERS = [
    "时间",
    "类型",
    "来源",
    "时长",
    "主要行为",
    "最高置信度",
    "识别次数",
    "意图摘要",
    "行为分布",
]


def export_records_csv(records: List[Dict], file_path: str | Path) -> int:
    """将记录导出为 Excel 友好的 UTF-8 BOM CSV，返回导出条数。"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADERS)
        for record in records:
            distribution = "；".join(
                f"{item['display']}×{item['count']}"
                for item in behavior_distribution(record)
            )
            writer.writerow(
                [
                    format_timestamp(record.get("timestamp"), "%Y-%m-%d %H:%M:%S"),
                    source_kind(record),
                    source_label(record),
                    format_duration(record.get("duration")),
                    dominant_behavior(record),
                    f"{top_confidence(record):.0%}",
                    detection_count(record),
                    record_summary(record, limit=200),
                    distribution or "—",
                ]
            )
            count += 1
    return count


def record_to_markdown(record: Dict) -> str:
    timestamp = format_timestamp(record.get("timestamp"))
    lines = [
        "# 宠物行为分析记录",
        "",
        f"- **分析时间**：{timestamp}",
        f"- **来源**：{source_label(record)}",
        f"- **时长**：{format_duration(record.get('duration'))}",
        f"- **识别次数**：{detection_count(record)}",
        f"- **主要行为**：{dominant_behavior(record)}",
        f"- **最高置信度**：{top_confidence(record):.0%}",
        "",
        "## 意图解读",
        "",
        record_summary(record, limit=500) or "—",
        "",
        "## 行为分布",
        "",
    ]

    distribution = behavior_distribution(record)
    if distribution:
        lines.extend(["| 行为 | 次数 | 占比 |", "| --- | ---: | ---: |"])
        for item in distribution:
            lines.append(
                f"| {item['display']} | {item['count']} | {item['percent']:.0%} |"
            )
    else:
        lines.append("未识别到有效行为。")

    behaviors = record.get("behaviors", []) or []
    lines.extend(["", "## 行为时间线", ""])
    if behaviors:
        lines.extend(["| 时间 | 行为 | 置信度 | 意图 |", "| --- | --- | ---: | --- |"])
        for item in behaviors:
            behavior = item.get("behavior", "")
            if behavior in {"", "idle", "error"}:
                continue
            display = BEHAVIOR_DISPLAY_NAMES.get(behavior, behavior)
            confidence = float(item.get("confidence", 0.0) or 0.0)
            moment = format_offset(item.get("timestamp", 0))
            intent = (item.get("intent", "") or "").replace("|", "／")
            lines.append(f"| {moment} | {display} | {confidence:.0%} | {intent} |")
    else:
        lines.append("暂无行为明细。")

    lines.extend(["", "---", "", "由宠物行为智能分析软件生成。"])
    return "\n".join(lines)


def export_record_markdown(record: Dict, file_path: str | Path) -> int:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = record_to_markdown(record)
    path.write_text(content, encoding="utf-8")
    return len(content)
