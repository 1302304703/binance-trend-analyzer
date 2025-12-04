#!/usr/bin/env python3
"""
Web服务入口
@author Reln Ding
"""

import os
import sys
import atexit
from flask import Flask, render_template, jsonify, request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cache import get_cache
from collector import get_klines
from indicators import calculate_all_indicators
from prediction_tracker import get_tracker
from config import REFRESH_INTERVAL, SYMBOLS, INTERVALS

app = Flask(__name__, template_folder='templates', static_folder='static')

# 获取全局缓存实例
cache = get_cache()

# 启动后台更新（每30秒更新一次）
cache.start_background_update(interval=30.0)

# 注册退出时停止后台线程
atexit.register(cache.stop_background_update)


@app.route('/')
def index():
    """首页"""
    return render_template('index.html', refresh_interval=REFRESH_INTERVAL)


@app.route('/chart')
def chart():
    """图表页面"""
    return render_template('chart.html', symbols=SYMBOLS, intervals=INTERVALS)


@app.route('/api/analysis')
def get_analysis():
    """获取分析数据API（使用缓存，秒级响应）"""
    try:
        # 从缓存获取分析结果
        results = cache.get_full_analysis()

        # 获取缓存统计
        cache_stats = cache.get_cache_stats()

        # 转换数据格式
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cache_age': round(cache_stats.get('analysis_age') or 0, 1),
            'symbols': {}
        }

        for symbol, intervals in results.items():
            data['symbols'][symbol] = {
                'intervals': {},
                'resonance': intervals.get('resonance', {})
            }

            for interval, interval_data in intervals.items():
                if interval == 'resonance':
                    continue

                pred = interval_data['prediction']

                # 获取交易建议
                advice = interval_data.get('trading_advice', {})

                # 简化数据
                data['symbols'][symbol]['intervals'][interval] = {
                    'current_price': interval_data['current_price'],
                    'stats_24h': interval_data['stats_24h'],
                    'prediction': {
                        'direction': pred['overall_direction'],
                        'confidence': pred['confidence'],
                        'score': pred['score'],
                        'volume_weight': pred.get('volume_weight', 1.0),
                        'change_percent': pred['linear_change_percent'],
                        'trend_strength': pred.get('trend_strength', {}),
                        'momentum': pred.get('momentum', {}),
                        'obv_divergence': pred.get('obv_divergence', {}),
                    },
                    'trading_advice': {
                        'action': advice.get('action', '观望'),
                        'action_en': advice.get('action_en', 'wait'),
                        'reason': advice.get('reason', ''),
                        'entry_price': advice.get('entry_price'),
                        'stop_loss': advice.get('stop_loss'),
                        'stop_loss_percent': advice.get('stop_loss_percent'),
                        'take_profit_1': advice.get('take_profit_1'),
                        'take_profit_2': advice.get('take_profit_2'),
                        'take_profit_3': advice.get('take_profit_3'),
                        'risk_reward_ratio': advice.get('risk_reward_ratio'),
                        'position_suggestion': advice.get('position_suggestion', ''),
                        'leverage_suggestion': advice.get('leverage_suggestion', ''),
                        'max_leverage': advice.get('max_leverage'),
                        'safe_leverage': advice.get('safe_leverage'),
                        'risk_level': advice.get('risk_level', '中'),
                        'key_points': advice.get('key_points', []),
                    },
                    'indicators': {
                        'rsi': interval_data['indicators'].get('RSI'),
                        'macd_hist': interval_data['indicators'].get('MACD_Hist')
                    },
                    'pattern': interval_data.get('pattern', ''),
                    'pattern_type': interval_data.get('pattern_type', ''),
                    'support': interval_data.get('nearest_support'),
                    'resistance': interval_data.get('nearest_resistance'),
                }

        return jsonify({'success': True, 'data': data})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/cache-stats')
def get_cache_stats():
    """获取缓存状态（用于调试）"""
    stats = cache.get_cache_stats()
    return jsonify({
        'success': True,
        'data': {
            'cached_items': stats['cached_items'],
            'analysis_age_seconds': round(stats['analysis_age'] or 0, 1),
            'is_updating': stats['is_updating'],
            'background_running': stats['background_running']
        }
    })


@app.route('/api/accuracy')
def get_accuracy():
    """获取预测准确率统计"""
    try:
        tracker = get_tracker()
        stats = tracker.get_accuracy_stats()
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


@app.route('/api/klines')
def get_klines_api():
    """获取K线数据API - 用于图表展示"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        interval = request.args.get('interval', '1h')
        limit = int(request.args.get('limit', 200))

        df = get_klines(symbol, interval, limit=limit)
        if df is None:
            return jsonify({'success': False, 'error': '获取数据失败'})

        # 计算指标
        df = calculate_all_indicators(df)

        # 转换为图表所需格式
        klines = []
        for _, row in df.iterrows():
            klines.append({
                'time': int(row['open_time'].timestamp()),
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume'],
            })

        # 提取MA数据
        ma_data = {
            'ma7': [{'time': int(row['open_time'].timestamp()), 'value': row['MA7']}
                    for _, row in df.iterrows() if not pd.isna(row.get('MA7'))],
            'ma25': [{'time': int(row['open_time'].timestamp()), 'value': row['MA25']}
                     for _, row in df.iterrows() if not pd.isna(row.get('MA25'))],
            'ma99': [{'time': int(row['open_time'].timestamp()), 'value': row['MA99']}
                     for _, row in df.iterrows() if not pd.isna(row.get('MA99'))],
        }

        # 提取布林带数据
        boll_data = {
            'upper': [{'time': int(row['open_time'].timestamp()), 'value': row['BOLL_Upper']}
                      for _, row in df.iterrows() if not pd.isna(row.get('BOLL_Upper'))],
            'middle': [{'time': int(row['open_time'].timestamp()), 'value': row['BOLL_Middle']}
                       for _, row in df.iterrows() if not pd.isna(row.get('BOLL_Middle'))],
            'lower': [{'time': int(row['open_time'].timestamp()), 'value': row['BOLL_Lower']}
                      for _, row in df.iterrows() if not pd.isna(row.get('BOLL_Lower'))],
        }

        # 提取形态数据
        patterns = []
        for _, row in df.iterrows():
            if row.get('pattern'):
                patterns.append({
                    'time': int(row['open_time'].timestamp()),
                    'pattern': row['pattern'],
                    'type': row['pattern_type'],
                    'price': row['high'] if row['pattern_type'] == 'bearish' else row['low']
                })

        # 支撑阻力位
        latest = df.iloc[-1]
        support_resistance = {
            'supports': latest.get('support_levels', []),
            'resistances': latest.get('resistance_levels', []),
            'nearest_support': latest.get('nearest_support'),
            'nearest_resistance': latest.get('nearest_resistance'),
        }

        return jsonify({
            'success': True,
            'data': {
                'klines': klines,
                'ma': ma_data,
                'boll': boll_data,
                'patterns': patterns,
                'support_resistance': support_resistance,
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()})


# 需要导入pandas
import pandas as pd


if __name__ == '__main__':
    print("启动Web服务: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
