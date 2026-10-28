"""历史记录管理模块 - SQLite 数据库操作

使用上下文管理器确保连接正确关闭，避免连接泄漏。
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager
import os

from utils.logger import get_logger


class _ConnectionManager:
    """连接上下文管理器，对内存数据库不关闭连接"""

    def __init__(self, conn: sqlite3.Connection, is_memory: bool):
        self.conn = conn
        self.is_memory = is_memory

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_memory:
            self.conn.close()
        return False


class RecordManager:
    def __init__(self, db_path: str = './data/records.db'):
        self._is_memory = db_path == ':memory:'
        if self._is_memory:
            self.db_path = ':memory:'
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, db_path.replace('./', ''))
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.logger = get_logger()
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _get_connection(self) -> _ConnectionManager:
        if self._is_memory:
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(':memory:')
                self._persistent_conn.row_factory = sqlite3.Row
            return _ConnectionManager(self._persistent_conn, is_memory=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return _ConnectionManager(conn, is_memory=False)

    def _init_database(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS behavior_records (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_path TEXT,
                        duration REAL,
                        behaviors TEXT,
                        summary TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
            self.logger.info(f"Database initialized at: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")

    def add_record(
        self,
        source_type: str,
        source_path: Optional[str] = None,
        duration: float = 0.0,
        behaviors: Optional[List[Dict]] = None,
        summary: str = ''
    ) -> str:
        """添加新的行为记录
        
        Args:
            source_type: 数据源类型（camera/video/image）
            source_path: 数据源路径
            duration: 分析时长（秒）
            behaviors: 行为列表
            summary: 行为摘要
            
        Returns:
            新记录的ID，失败返回空字符串
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        behaviors_json = json.dumps(behaviors or [], ensure_ascii=False)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO behavior_records
                    (id, timestamp, source_type, source_path, duration, behaviors, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (record_id, timestamp, source_type, source_path, duration, behaviors_json, summary))
                conn.commit()
            self.logger.info(f"Record added: {record_id}")
            return record_id
        except Exception as e:
            self.logger.error(f"Error adding record: {str(e)}")
            return ''

    def update_record(
        self,
        record_id: str,
        behaviors: List[Dict],
        summary: str,
        duration: Optional[float] = None,
    ):
        """更新记录内容，可选同时写入分析时长。"""
        try:
            behaviors_json = json.dumps(behaviors, ensure_ascii=False)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if duration is None:
                    cursor.execute('''
                        UPDATE behavior_records
                        SET behaviors = ?, summary = ?
                        WHERE id = ?
                    ''', (behaviors_json, summary, record_id))
                else:
                    cursor.execute('''
                        UPDATE behavior_records
                        SET behaviors = ?, summary = ?, duration = ?
                        WHERE id = ?
                    ''', (behaviors_json, summary, duration, record_id))
                conn.commit()
            self.logger.info(f"Record updated: {record_id}")
        except Exception as e:
            self.logger.error(f"Error updating record: {str(e)}")

    def get_record(self, record_id: str) -> Optional[Dict]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, timestamp, source_type, source_path, duration, behaviors, summary, created_at
                    FROM behavior_records WHERE id = ?
                ''', (record_id,))
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0], 'timestamp': row[1], 'source_type': row[2],
                        'source_path': row[3], 'duration': row[4],
                        'behaviors': json.loads(row[5]) if row[5] else [],
                        'summary': row[6], 'created_at': row[7]
                    }
            return None
        except Exception as e:
            self.logger.error(f"Error getting record: {str(e)}")
            return None

    def get_all_records(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, timestamp, source_type, source_path, duration, behaviors, summary, created_at
                    FROM behavior_records ORDER BY timestamp DESC LIMIT ? OFFSET ?
                ''', (limit, offset))
                rows = cursor.fetchall()
            return [{
                'id': row[0], 'timestamp': row[1], 'source_type': row[2],
                'source_path': row[3], 'duration': row[4],
                'behaviors': json.loads(row[5]) if row[5] else [],
                'summary': row[6], 'created_at': row[7]
            } for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting all records: {str(e)}")
            return []

    def delete_record(self, record_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM behavior_records WHERE id = ?', (record_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
            if deleted:
                self.logger.info(f"Record deleted: {record_id}")
            return deleted
        except Exception as e:
            self.logger.error(f"Error deleting record: {str(e)}")
            return False

    def clear_all_records(self):
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM behavior_records')
                conn.commit()
            self.logger.info("All records cleared")
        except Exception as e:
            self.logger.error(f"Error clearing records: {str(e)}")

    def get_records_count(self) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM behavior_records')
                count = cursor.fetchone()[0]
            return count
        except Exception as e:
            self.logger.error(f"Error getting records count: {str(e)}")
            return 0

    def search_records(self, keyword: str, limit: int = 50) -> List[Dict]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, timestamp, source_type, source_path, duration, behaviors, summary, created_at
                    FROM behavior_records
                    WHERE summary LIKE ? OR source_path LIKE ?
                    ORDER BY timestamp DESC LIMIT ?
                ''', (f'%{keyword}%', f'%{keyword}%', limit))
                rows = cursor.fetchall()
            return [{
                'id': row[0], 'timestamp': row[1], 'source_type': row[2],
                'source_path': row[3], 'duration': row[4],
                'behaviors': json.loads(row[5]) if row[5] else [],
                'summary': row[6], 'created_at': row[7]
            } for row in rows]
        except Exception as e:
            self.logger.error(f"Error searching records: {str(e)}")
            return []
