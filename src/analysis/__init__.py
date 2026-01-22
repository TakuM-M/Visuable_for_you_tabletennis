"""
analysis パッケージ

プレイヤーと卓球台の位置関係など、検出結果の分析を行うモジュール
"""

from .data_classes import PlayerTableRelation
from .player_table_analyzer import PlayerTableAnalyzer

__all__ = [
    'PlayerTableRelation',
    'PlayerTableAnalyzer',
]
