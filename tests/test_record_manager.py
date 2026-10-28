"""记录管理器单元测试"""

import pytest
import tempfile
import os
import gc


@pytest.fixture
def record_manager():
    """提供 RecordManager 实例，使用内存数据库避免文件锁定问题"""
    from core.record_manager import RecordManager
    manager = RecordManager(':memory:')
    yield manager
    gc.collect()


class TestRecordManager:
    """测试记录管理器"""

    def test_add_record(self, record_manager):
        """测试添加记录"""
        record_id = record_manager.add_record(
            source_type='camera',
            source_path='Camera 0',
            duration=10.0,
            behaviors=[{'behavior': 'sitting', 'confidence': 0.8}],
            summary='Test record'
        )
        assert record_id != ''

    def test_get_record(self, record_manager):
        """测试获取记录"""
        record_id = record_manager.add_record(
            source_type='camera',
            source_path='Camera 0',
            duration=10.0,
            behaviors=[{'behavior': 'sitting', 'confidence': 0.8}],
            summary='Test record'
        )

        record = record_manager.get_record(record_id)
        assert record is not None
        assert record['source_type'] == 'camera'
        assert record['summary'] == 'Test record'

    def test_update_record(self, record_manager):
        """测试更新记录"""
        record_id = record_manager.add_record(
            source_type='camera',
            source_path='Camera 0',
            duration=10.0
        )

        record_manager.update_record(
            record_id,
            behaviors=[{'behavior': 'walking', 'confidence': 0.9}],
            summary='Updated record'
        )

        record = record_manager.get_record(record_id)
        assert record['summary'] == 'Updated record'

    def test_delete_record(self, record_manager):
        """测试删除记录"""
        record_id = record_manager.add_record(
            source_type='camera',
            source_path='Camera 0',
            duration=10.0
        )

        result = record_manager.delete_record(record_id)
        assert result is True

        record = record_manager.get_record(record_id)
        assert record is None

    def test_get_all_records(self, record_manager):
        """测试获取所有记录"""
        record_manager.add_record(source_type='camera', source_path='Camera 0')
        record_manager.add_record(source_type='video', source_path='test.mp4')

        records = record_manager.get_all_records()
        assert len(records) == 2

    def test_get_records_count(self, record_manager):
        """测试获取记录数量"""
        record_manager.add_record(source_type='camera', source_path='Camera 0')
        record_manager.add_record(source_type='video', source_path='test.mp4')

        count = record_manager.get_records_count()
        assert count == 2

    def test_clear_all_records(self, record_manager):
        """测试清空所有记录"""
        record_manager.add_record(source_type='camera', source_path='Camera 0')
        record_manager.add_record(source_type='video', source_path='test.mp4')

        record_manager.clear_all_records()

        count = record_manager.get_records_count()
        assert count == 0

    def test_search_records(self, record_manager):
        """测试搜索记录"""
        record_manager.add_record(
            source_type='camera',
            source_path='Camera 0',
            summary='Test record 1'
        )
        record_manager.add_record(
            source_type='video',
            source_path='test.mp4',
            summary='Another record'
        )

        results = record_manager.search_records('Test')
        assert len(results) >= 1

    def test_get_nonexistent_record(self, record_manager):
        """测试获取不存在的记录"""
        record = record_manager.get_record('nonexistent_id')
        assert record is None

    def test_delete_nonexistent_record(self, record_manager):
        """测试删除不存在的记录"""
        result = record_manager.delete_record('nonexistent_id')
        assert result is False