"""历史记录导出功能的单元测试。"""

from __future__ import annotations

import csv
from pathlib import Path

from utils.exporters import (
    CSV_HEADERS,
    export_record_markdown,
    export_records_csv,
    record_to_markdown,
)


def _record():
    return {
        "id": "rec-1",
        "timestamp": "2026-10-27T14:59:01",
        "source_type": "video",
        "source_path": r"D:\videos\living_room_cat.mp4",
        "duration": 42.0,
        "summary": "猫咪在客厅里伸展身体，状态放松。",
        "behaviors": [
            {"timestamp": 0.0, "behavior": "stretching", "confidence": 0.92, "intent": "舒展身体"},
            {"timestamp": 6.4, "behavior": "walking", "confidence": 0.71, "intent": "巡视 | 领地"},
            {"timestamp": 12.8, "behavior": "idle", "confidence": 0.30, "intent": ""},
        ],
    }


class TestCsvExport:
    def test_header_and_row(self, tmp_path: Path):
        target = tmp_path / "out" / "records.csv"
        count = export_records_csv([_record()], target)

        assert count == 1
        assert target.exists()
        raw = target.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")  # Excel 友好的 UTF-8 BOM

        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))

        assert rows[0] == CSV_HEADERS
        assert rows[1][0] == "2026-10-27 14:59:01"
        assert rows[1][1] == "视频"
        assert rows[1][2] == "视频 · living_room_cat.mp4"
        assert rows[1][3] == "00:42"
        assert rows[1][4] in {"伸懒腰", "行走"}
        assert rows[1][5] == "92%"
        assert rows[1][6] == "2"
        assert rows[1][7] == "猫咪在客厅里伸展身体，状态放松。"
        assert "行走×1" in rows[1][8]

    def test_empty_list_still_writes_header(self, tmp_path: Path):
        target = tmp_path / "empty.csv"
        assert export_records_csv([], target) == 0
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows == [CSV_HEADERS]


class TestMarkdownExport:
    def test_timeline_starts_at_zero(self):
        markdown = record_to_markdown(_record())
        assert "| 00:00 | 伸懒腰 | 92% | 舒展身体 |" in markdown
        assert "| 00:06 | 行走 | 71% | 巡视 ／ 领地 |" in markdown
        assert "发呆" not in markdown  # idle 行为不计入时间线

    def test_metadata_and_distribution(self):
        markdown = record_to_markdown(_record())
        assert markdown.startswith("# 宠物行为分析记录")
        assert "- **来源**：视频 · living_room_cat.mp4" in markdown
        assert "- **时长**：00:42" in markdown
        assert "- **识别次数**：2" in markdown
        assert "| 伸懒腰 | 1 | 50% |" in markdown
        assert "| 行走 | 1 | 50% |" in markdown

    def test_empty_behaviors_render_placeholder(self):
        record = _record()
        record["behaviors"] = []
        record["summary"] = ""
        markdown = record_to_markdown(record)
        assert "未识别到有效行为。" in markdown
        assert "暂无行为明细。" in markdown

    def test_writes_file_and_returns_length(self, tmp_path: Path):
        target = tmp_path / "report.md"
        length = export_record_markdown(_record(), target)
        content = target.read_text(encoding="utf-8")
        assert length == len(content)
        assert "宠物行为分析记录" in content
