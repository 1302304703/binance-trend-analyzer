"""
预测准确率追踪模块 - 使用 JSON 文件存储
@author Reln Ding
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

# 各周期的验证时间（分钟）
VERIFY_PERIODS = {
    '5m': 5,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}

# 预测方向映射
DIRECTION_MAP = {
    '强烈看涨': 'bullish',
    '看涨': 'bullish',
    '偏多': 'bullish',
    '强烈看跌': 'bearish',
    '看跌': 'bearish',
    '偏空': 'bearish',
    '震荡观望': 'neutral',
}

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predictions.json')


class PredictionTracker:
    """预测准确率追踪器（JSON版本）"""

    def __init__(self):
        self._lock = threading.RLock()
        self._predictions: List[Dict] = []
        self._load_predictions()

    def _load_predictions(self):
        """从文件加载预测数据"""
        if os.path.exists(PREDICTIONS_FILE):
            try:
                with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
                    self._predictions = json.load(f)
            except Exception as e:
                print(f"[WARN] 加载预测数据失败: {e}")
                self._predictions = []
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            self._predictions = []

    def _save_predictions(self):
        """保存预测数据到文件"""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._predictions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[ERROR] 保存预测数据失败: {e}")

    def record_prediction(self, symbol: str, interval: str,
                          direction: str, confidence: str,
                          score: float, price: float):
        """
        记录一条预测

        Args:
            symbol: 交易对
            interval: 周期
            direction: 预测方向
            confidence: 置信度
            score: 评分
            price: 当前价格
        """
        now = time.time()

        # 标准化方向
        direction_normalized = DIRECTION_MAP.get(direction, 'neutral')

        prediction = {
            'id': f"{symbol}_{interval}_{int(now)}",
            'symbol': symbol,
            'interval': interval,
            'direction': direction,
            'direction_normalized': direction_normalized,
            'confidence': confidence,
            'score': score,
            'price_at_prediction': price,
            'timestamp': now,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'verified': False,
            'actual_direction': None,
            'price_at_verify': None,
            'price_change_percent': None,
            'is_correct': None
        }

        with self._lock:
            # 检查是否已存在相同的预测（同一symbol+interval在短时间内）
            exists = False
            for p in self._predictions:
                if (p['symbol'] == symbol and
                    p['interval'] == interval and
                    abs(p['timestamp'] - now) < 60):  # 1分钟内不重复记录
                    exists = True
                    break

            if not exists:
                self._predictions.append(prediction)
                self._save_predictions()

    def verify_predictions(self, current_prices: Dict[str, float]):
        """
        验证历史预测

        Args:
            current_prices: {symbol: current_price}
        """
        now = time.time()
        updated = False

        with self._lock:
            for pred in self._predictions:
                if pred['verified']:
                    continue

                symbol = pred['symbol']
                interval = pred['interval']
                timestamp = pred['timestamp']

                # 检查是否到了验证时间
                verify_minutes = VERIFY_PERIODS.get(interval, 60)
                verify_time = timestamp + (verify_minutes * 60)

                if now < verify_time:
                    continue

                # 获取当前价格
                current_price = current_prices.get(symbol)
                if current_price is None:
                    continue

                # 计算价格变化
                price_at_prediction = pred['price_at_prediction']
                price_change = current_price - price_at_prediction
                price_change_percent = (price_change / price_at_prediction) * 100

                # 判断实际方向
                if price_change_percent > 0.1:
                    actual_direction = 'bullish'
                elif price_change_percent < -0.1:
                    actual_direction = 'bearish'
                else:
                    actual_direction = 'neutral'

                # 判断预测是否正确
                predicted_direction = pred['direction_normalized']
                is_correct = self._check_prediction_correct(
                    predicted_direction, actual_direction, price_change_percent
                )

                # 更新预测记录
                pred['verified'] = True
                pred['actual_direction'] = actual_direction
                pred['price_at_verify'] = current_price
                pred['price_change_percent'] = round(price_change_percent, 4)
                pred['is_correct'] = is_correct
                pred['verify_datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                updated = True

            if updated:
                self._save_predictions()

    def _check_prediction_correct(self, predicted: str, actual: str,
                                   change_percent: float) -> bool:
        """判断预测是否正确"""
        if predicted == 'neutral':
            return abs(change_percent) < 0.5
        return predicted == actual

    def get_accuracy_stats(self) -> Dict:
        """获取准确率统计"""
        with self._lock:
            total = len(self._predictions)
            verified = [p for p in self._predictions if p['verified']]
            verified_count = len(verified)
            correct_count = len([p for p in verified if p['is_correct']])

            # 按周期统计
            by_interval = {}
            for p in verified:
                interval = p['interval']
                if interval not in by_interval:
                    by_interval[interval] = {'total': 0, 'correct': 0}
                by_interval[interval]['total'] += 1
                if p['is_correct']:
                    by_interval[interval]['correct'] += 1

            # 计算各周期准确率
            for interval, stats in by_interval.items():
                if stats['total'] > 0:
                    stats['accuracy'] = round(stats['correct'] / stats['total'] * 100, 1)
                else:
                    stats['accuracy'] = 0

            # 按置信度统计
            by_confidence = {}
            for p in verified:
                conf = p.get('confidence', '中')
                if conf not in by_confidence:
                    by_confidence[conf] = {'total': 0, 'correct': 0}
                by_confidence[conf]['total'] += 1
                if p['is_correct']:
                    by_confidence[conf]['correct'] += 1

            # 计算各置信度准确率
            for conf, stats in by_confidence.items():
                if stats['total'] > 0:
                    stats['accuracy'] = round(stats['correct'] / stats['total'] * 100, 1)
                else:
                    stats['accuracy'] = 0

            overall_accuracy = round(correct_count / verified_count * 100, 1) if verified_count > 0 else 0

            return {
                'total_predictions': total,
                'verified_count': verified_count,
                'pending_count': total - verified_count,
                'correct_count': correct_count,
                'overall_accuracy': overall_accuracy,
                'by_interval': by_interval,
                'by_confidence': by_confidence
            }

    def get_interval_accuracy(self, symbol: str, interval: str) -> Optional[Dict]:
        """获取特定交易对和周期的准确率"""
        stats = self.get_accuracy_stats()
        by_interval = stats.get('by_interval', {})
        return by_interval.get(interval)


# 全局实例
_tracker_instance: Optional[PredictionTracker] = None


def get_tracker() -> PredictionTracker:
    """获取全局追踪器实例"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PredictionTracker()
    return _tracker_instance
