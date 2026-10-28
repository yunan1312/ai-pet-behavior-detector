"""记录展示纯函数的单元测试。"""

from __future__ import annotations

import pytest

from utils.record_view import (
    behavior_counts,
    behavior_distribution,
    detection_count,
    dominant_behavior,
    format_duration,
    format_offset,
    format_timestamp,
    parse_timestamp,
    record_summary,
    source_kind,
    source_label,
    top_confidence,
)


def _record(**overrides):
    record = {
        "id": "rec-1",
        "timestamp": "2026-10-27T14:59:01",
        "source_type": "video",
        "source_path": r"D:\videos\living_room_cat.mp4",
        "duration": 42.0,
        "summary": "猫咪在客厅里伸展身体，状态放松。",
        "behaviors": [
            {"timestamp": 0.0, "behavior": "stretching", "confidence": 0.92, "intent": "舒展身体"},
            {"timestamp": 6.4, "behavior": "walking", "confidence": 0.71, "intent": "巡视领地"},
            {"timestamp": 12.8, "behavior": "walking", "confidence": 0.66, "intent": "巡视领地"},
            {"timestamp": 13.0, "behavior": "idle", "confidence": 0.30, "intent": ""},
        ],
    }
    record.update(overrides)
    return record


class TestFormatOffset:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "00:00"),
            (0.0, "00:00"),
            (6.4, "00:06"),
            (59.6, "01:00"),
            (65, "01:05"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (None, "--"),
            (-1, "--"),
            ("bad", "--"),
        ],
    )
    def test_format_offset(self, value, expected):
        assert format_offset(value) == expected

    def test_missing_offset_is_not_confused_with_zero(self):
        assert format_offset(None) == "--"
        assert format_offset(0) == "00:00"


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(42, "00:42"), (3725, "1:02:05"), (0, "--"), (None, "--"), ("bad", "--")],
    )
    def test_format_duration(self, value, expected):
        assert format_duration(value) == expected


class TestSourceLabels:
    def test_video_source_includes_filename(self):
        record = _record()
        assert source_kind(record) == "视频"
        assert source_label(record) == "视频 · living_room_cat.mp4"

    def test_camera_source_hides_placeholder_path(self):
        record = _record(source_type="camera", source_path="Camera 0")
        assert source_label(record) == "摄像头"

    def test_unknown_source(self):
        record = _record(source_type="weird", source_path="")
        assert source_kind(record) == "其他"
        assert source_label(record) == "weird"


class TestBehaviorSummary:
    def test_ignores_idle_and_error(self):
        record = _record()
        assert detection_count(record) == 3
        assert behavior_counts(record) == [("walking", 2), ("stretching", 1)]
        assert dominant_behavior(record) == "行走"
        assert top_confidence(record) == pytest.approx(0.92)

    def test_distribution_percentages(self):
        distribution = behavior_distribution(_record())
        assert [item["count"] for item in distribution] == [2, 1]
        assert distribution[0]["percent"] == pytest.approx(2 / 3)
        assert distribution[1]["percent"] == pytest.approx(1 / 3)

    def test_empty_record(self):
        record = _record(behaviors=[], summary="")
        assert detection_count(record) == 0
        assert dominant_behavior(record) == "—"
        assert top_confidence(record) == 0.0
        assert behavior_distribution(record) == []
        assert record_summary(record) == "—"


class TestSummaryAndTimestamp:
    def test_summary_falls_back_to_first_intent(self):
        record = _record(summary="")
        assert record_summary(record) == "舒展身体"

    def test_summary_truncates_with_ellipsis(self):
        record = _record(summary="猫" * 100)
        summary = record_summary(record, limit=10)
        assert len(summary) == 10
        assert summary.endswith("…")

    def test_timestamp_parsing(self):
        assert parse_timestamp("2026-10-27T14:59:01").year == 2026
        assert parse_timestamp("not-a-date") is None
        assert format_timestamp("bad-value") == "bad-value"
        assert format_timestamp(None) == "--"
        assert format_timestamp("2026-10-27T14:59:01") == "2026-10-27 14:59"
