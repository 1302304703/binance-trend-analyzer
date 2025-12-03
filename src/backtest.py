"""
回测系统 - 验证预测策略的历史准确率
@author Reln Ding
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from tabulate import tabulate

from collector import get_klines
from indicators import calculate_all_indicators
from predictor import TrendPredictor
from config import SYMBOLS, INTERVALS


class Backtester:
    """回测器"""

    def __init__(self, lookback_periods: int = 100, prediction_periods: int = 5):
        """
        初始化回测器

        Args:
            lookback_periods: 用于计算指标的历史K线数量
            prediction_periods: 预测未来K线数量
        """
        self.lookback_periods = lookback_periods
        self.prediction_periods = prediction_periods
        self.results = {}

    def run_backtest(self, symbol: str, interval: str, test_periods: int = 100) -> Dict:
        """
        运行单个交易对的回测

        Args:
            symbol: 交易对
            interval: K线周期
            test_periods: 回测的K线数量

        Returns:
            回测结果
        """
        # 获取足够的历史数据
        total_needed = self.lookback_periods + test_periods + self.prediction_periods + 50
        df = get_klines(symbol, interval, limit=min(total_needed, 1000))

        if df is None or len(df) < self.lookback_periods + test_periods:
            return {'error': '数据不足'}

        predictions = []
        actuals = []
        details = []

        # 从lookback_periods位置开始回测
        start_idx = self.lookback_periods + 50  # 留出足够的数据计算指标

        for i in range(start_idx, len(df) - self.prediction_periods):
            # 截取历史数据用于预测
            historical_df = df.iloc[:i + 1].copy()

            # 计算指标
            historical_df = calculate_all_indicators(historical_df)

            # 进行预测
            predictor = TrendPredictor(historical_df)
            prediction = predictor.get_comprehensive_prediction()

            # 获取预测结果
            pred_direction = prediction['overall_direction']
            pred_score = prediction['score']
            confidence = prediction['confidence']

            # 计算实际结果（未来N根K线的价格变化）
            current_price = df.iloc[i]['close']
            future_price = df.iloc[i + self.prediction_periods]['close']
            actual_change = ((future_price - current_price) / current_price) * 100

            # 判断实际方向
            if actual_change > 0.3:
                actual_direction = "上涨"
            elif actual_change < -0.3:
                actual_direction = "下跌"
            else:
                actual_direction = "横盘"

            # 判断预测是否正确
            pred_bullish = '涨' in pred_direction or '多' in pred_direction
            pred_bearish = '跌' in pred_direction or '空' in pred_direction
            actual_bullish = actual_direction == "上涨"
            actual_bearish = actual_direction == "下跌"

            if pred_bullish and actual_bullish:
                is_correct = True
            elif pred_bearish and actual_bearish:
                is_correct = True
            elif not pred_bullish and not pred_bearish and actual_direction == "横盘":
                is_correct = True
            elif pred_bullish and actual_bearish:
                is_correct = False
            elif pred_bearish and actual_bullish:
                is_correct = False
            else:
                # 预测观望但实际有方向，算部分正确
                is_correct = None

            predictions.append({
                'direction': pred_direction,
                'score': pred_score,
                'confidence': confidence
            })
            actuals.append({
                'direction': actual_direction,
                'change': actual_change
            })
            details.append({
                'time': df.iloc[i]['open_time'],
                'price': current_price,
                'pred_direction': pred_direction,
                'pred_score': pred_score,
                'confidence': confidence,
                'actual_direction': actual_direction,
                'actual_change': actual_change,
                'is_correct': is_correct
            })

        # 计算统计结果
        stats = self._calculate_stats(details)

        return {
            'symbol': symbol,
            'interval': interval,
            'test_periods': len(details),
            'stats': stats,
            'details': details
        }

    def _calculate_stats(self, details: List[Dict]) -> Dict:
        """
        计算回测统计数据

        Args:
            details: 详细回测记录

        Returns:
            统计结果
        """
        total = len(details)
        if total == 0:
            return {}

        correct = sum(1 for d in details if d['is_correct'] is True)
        incorrect = sum(1 for d in details if d['is_correct'] is False)
        neutral = sum(1 for d in details if d['is_correct'] is None)

        # 按置信度分组统计
        confidence_stats = {}
        for conf in ['极高', '高', '中高', '中', '低']:
            conf_details = [d for d in details if d['confidence'] == conf]
            if conf_details:
                conf_correct = sum(1 for d in conf_details if d['is_correct'] is True)
                conf_total = len(conf_details)
                confidence_stats[conf] = {
                    'total': conf_total,
                    'correct': conf_correct,
                    'accuracy': conf_correct / conf_total * 100 if conf_total > 0 else 0
                }

        # 按预测方向分组统计
        direction_stats = {}
        for direction_type in ['看涨', '看跌', '观望']:
            dir_details = [d for d in details if direction_type in d['pred_direction'] or
                          (direction_type == '观望' and '观望' in d['pred_direction'])]
            if dir_details:
                dir_correct = sum(1 for d in dir_details if d['is_correct'] is True)
                dir_total = len(dir_details)
                direction_stats[direction_type] = {
                    'total': dir_total,
                    'correct': dir_correct,
                    'accuracy': dir_correct / dir_total * 100 if dir_total > 0 else 0
                }

        # 计算盈利能力（假设按预测方向操作）
        profit_stats = self._calculate_profit_stats(details)

        return {
            'total': total,
            'correct': correct,
            'incorrect': incorrect,
            'neutral': neutral,
            'accuracy': correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0,
            'confidence_stats': confidence_stats,
            'direction_stats': direction_stats,
            'profit_stats': profit_stats
        }

    def _calculate_profit_stats(self, details: List[Dict]) -> Dict:
        """
        计算假设按预测操作的盈亏统计

        Args:
            details: 详细回测记录

        Returns:
            盈亏统计
        """
        total_profit = 0
        trades = 0
        wins = 0
        losses = 0

        for d in details:
            pred_direction = d['pred_direction']
            actual_change = d['actual_change']

            # 只统计有明确方向的预测
            if '涨' in pred_direction or '多' in pred_direction:
                # 做多
                profit = actual_change
                trades += 1
                total_profit += profit
                if profit > 0:
                    wins += 1
                else:
                    losses += 1
            elif '跌' in pred_direction or '空' in pred_direction:
                # 做空
                profit = -actual_change
                trades += 1
                total_profit += profit
                if profit > 0:
                    wins += 1
                else:
                    losses += 1

        return {
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': wins / trades * 100 if trades > 0 else 0,
            'total_profit_percent': total_profit,
            'avg_profit_per_trade': total_profit / trades if trades > 0 else 0
        }

    def run_full_backtest(self, test_periods: int = 100) -> Dict:
        """
        对所有配置的交易对和周期运行回测

        Args:
            test_periods: 每个周期回测的K线数量

        Returns:
            完整回测结果
        """
        results = {}
        for symbol in SYMBOLS:
            results[symbol] = {}
            for interval in INTERVALS:
                print(f"回测 {symbol} {interval}...")
                result = self.run_backtest(symbol, interval, test_periods)
                results[symbol][interval] = result

        self.results = results
        return results

    def print_report(self, results: Dict = None):
        """
        打印回测报告

        Args:
            results: 回测结果，如果为None则使用self.results
        """
        if results is None:
            results = self.results

        print("\n" + "=" * 80)
        print("  回测报告 - " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("=" * 80)

        summary_data = []

        for symbol, intervals in results.items():
            print(f"\n{'─' * 80}")
            print(f"  {symbol}")
            print(f"{'─' * 80}")

            for interval, data in intervals.items():
                if 'error' in data:
                    print(f"  {interval}: {data['error']}")
                    continue

                stats = data['stats']
                profit = stats['profit_stats']

                print(f"\n  ▶ 周期: {interval}")
                print(f"  ├─ 测试样本: {stats['total']} 次预测")
                print(f"  ├─ 预测准确率: {stats['accuracy']:.1f}%")
                print(f"  │  ├─ 正确: {stats['correct']}")
                print(f"  │  ├─ 错误: {stats['incorrect']}")
                print(f"  │  └─ 中性: {stats['neutral']}")
                print(f"  │")
                print(f"  ├─ 交易统计:")
                print(f"  │  ├─ 总交易次数: {profit['trades']}")
                print(f"  │  ├─ 胜率: {profit['win_rate']:.1f}%")
                print(f"  │  ├─ 累计收益: {profit['total_profit_percent']:.2f}%")
                print(f"  │  └─ 平均单次收益: {profit['avg_profit_per_trade']:.3f}%")

                # 置信度分析
                print(f"  │")
                print(f"  └─ 置信度分析:")
                for conf, conf_stats in stats['confidence_stats'].items():
                    print(f"     ├─ {conf}: {conf_stats['accuracy']:.1f}% ({conf_stats['correct']}/{conf_stats['total']})")

                summary_data.append({
                    '交易对': symbol,
                    '周期': interval,
                    '准确率': f"{stats['accuracy']:.1f}%",
                    '胜率': f"{profit['win_rate']:.1f}%",
                    '累计收益': f"{profit['total_profit_percent']:.2f}%",
                    '样本数': stats['total']
                })

        # 汇总表格
        print("\n" + "=" * 80)
        print("  汇总")
        print("=" * 80)
        print(tabulate(summary_data, headers='keys', tablefmt='grid'))

        # 计算总体统计
        total_accuracy = []
        total_win_rate = []
        total_profit = []

        for symbol, intervals in results.items():
            for interval, data in intervals.items():
                if 'error' not in data:
                    stats = data['stats']
                    if stats['accuracy'] > 0:
                        total_accuracy.append(stats['accuracy'])
                    profit = stats['profit_stats']
                    if profit['trades'] > 0:
                        total_win_rate.append(profit['win_rate'])
                        total_profit.append(profit['total_profit_percent'])

        if total_accuracy:
            print(f"\n  总体平均准确率: {np.mean(total_accuracy):.1f}%")
            print(f"  总体平均胜率: {np.mean(total_win_rate):.1f}%")
            print(f"  总体平均累计收益: {np.mean(total_profit):.2f}%")

        print("\n" + "=" * 80)
        print("  提示: 回测结果仅供参考，不代表未来表现")
        print("=" * 80 + "\n")


def main():
    """运行回测"""
    print("开始回测...")
    backtester = Backtester(lookback_periods=100, prediction_periods=5)

    # 运行完整回测
    results = backtester.run_full_backtest(test_periods=100)

    # 打印报告
    backtester.print_report(results)


if __name__ == '__main__':
    main()
