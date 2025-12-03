#!/usr/bin/env python3
"""
Web服务入口
@author Reln Ding
"""

import os
import sys
from flask import Flask, render_template, jsonify, request
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import TrendAnalyzer
from collector import get_klines
from indicators import calculate_all_indicators
from config import REFRESH_INTERVAL, SYMBOLS, INTERVALS

app = Flask(__name__, template_folder='templates', static_folder='static')
analyzer = TrendAnalyzer()


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
    """获取分析数据API"""
    try:
        results = analyzer.analyze_all()

        # 转换数据格式
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        return jsonify({'success': False, 'error': str(e)})


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
