"""
数据缓存模块 - 并行请求、内存缓存、后台预加载
@author Reln Ding
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import pandas as pd

from collector import get_klines, get_ticker_price, get_24h_stats
from indicators import calculate_all_indicators
from predictor import TrendPredictor
from prediction_tracker import get_tracker
from feedback_learning import get_online_learner
from config import SYMBOLS, INTERVALS, KLINE_LIMIT


# 减少K线数量以提升速度（300根足够分析）
OPTIMIZED_KLINE_LIMIT = 300


class DataCache:
    """数据缓存管理器"""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}  # 缓存数据
        self._cache_time: Dict[str, float] = {}  # 缓存时间戳
        self._lock = threading.RLock()  # 线程锁
        self._analysis_cache: Dict = {}  # 分析结果缓存
        self._analysis_time: float = 0
        self._updating = False  # 是否正在更新
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = False

    def _get_cache_key(self, symbol: str, interval: str) -> str:
        """生成缓存键"""
        return f"{symbol}_{interval}"

    def get_klines_cached(self, symbol: str, interval: str, max_age: float = 30.0) -> Optional[pd.DataFrame]:
        """
        获取缓存的K线数据

        Args:
            symbol: 交易对
            interval: K线周期
            max_age: 缓存最大有效期（秒）

        Returns:
            DataFrame或None
        """
        key = self._get_cache_key(symbol, interval)

        with self._lock:
            if key in self._cache:
                age = time.time() - self._cache_time.get(key, 0)
                if age < max_age:
                    return self._cache[key].get('klines')

        return None

    def set_klines_cache(self, symbol: str, interval: str, df: pd.DataFrame):
        """设置K线缓存"""
        key = self._get_cache_key(symbol, interval)
        with self._lock:
            if key not in self._cache:
                self._cache[key] = {}
            self._cache[key]['klines'] = df
            self._cache_time[key] = time.time()

    def fetch_single(self, symbol: str, interval: str) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """
        获取单个交易对的K线数据（用于并行）

        Returns:
            (symbol, interval, DataFrame)
        """
        # 先检查缓存
        cached = self.get_klines_cached(symbol, interval, max_age=25.0)
        if cached is not None:
            return (symbol, interval, cached)

        # 获取新数据
        df = get_klines(symbol, interval, limit=OPTIMIZED_KLINE_LIMIT)
        if df is not None:
            # 计算指标
            df = calculate_all_indicators(df)
            self.set_klines_cache(symbol, interval, df)

        return (symbol, interval, df)

    def fetch_all_parallel(self, symbols: List[str] = None, intervals: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        并行获取所有数据

        Args:
            symbols: 交易对列表
            intervals: 周期列表

        Returns:
            {symbol: {interval: DataFrame}}
        """
        symbols = symbols or SYMBOLS
        intervals = intervals or INTERVALS

        results: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in symbols}

        # 构建任务列表
        tasks = [(s, i) for s in symbols for i in intervals]

        # 并行执行
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.fetch_single, s, i): (s, i)
                for s, i in tasks
            }

            for future in as_completed(futures):
                try:
                    symbol, interval, df = future.result()
                    if df is not None:
                        results[symbol][interval] = df
                except Exception as e:
                    s, i = futures[future]
                    print(f"[ERROR] 并行获取失败 {s} {i}: {e}")

        return results

    def fetch_prices_parallel(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """
        并行获取价格和24h统计

        Returns:
            {symbol: {'price': float, 'stats_24h': dict}}
        """
        symbols = symbols or SYMBOLS
        results = {}

        def fetch_symbol_info(symbol: str) -> Tuple[str, float, dict]:
            price = get_ticker_price(symbol)
            stats = get_24h_stats(symbol)
            return (symbol, price, stats)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch_symbol_info, s) for s in symbols]

            for future in as_completed(futures):
                try:
                    symbol, price, stats = future.result()
                    results[symbol] = {
                        'price': price,
                        'stats_24h': stats
                    }
                except Exception as e:
                    print(f"[ERROR] 获取价格失败: {e}")

        return results

    def get_full_analysis(self, force_refresh: bool = False) -> Dict:
        """
        获取完整分析结果（使用缓存）

        Args:
            force_refresh: 是否强制刷新

        Returns:
            分析结果字典
        """
        # 检查分析缓存（有效期20秒）
        if not force_refresh and self._analysis_cache:
            age = time.time() - self._analysis_time
            if age < 20:
                return self._analysis_cache

        # 标记正在更新
        if self._updating:
            # 如果正在更新，返回旧缓存
            return self._analysis_cache or {}

        self._updating = True

        try:
            start_time = time.time()

            # 并行获取所有K线数据
            all_klines = self.fetch_all_parallel()

            # 并行获取价格信息
            all_prices = self.fetch_prices_parallel()

            # 构建分析结果
            results = {}
            for symbol in SYMBOLS:
                results[symbol] = {}
                symbol_klines = all_klines.get(symbol, {})
                price_info = all_prices.get(symbol, {})

                for interval in INTERVALS:
                    df = symbol_klines.get(interval)
                    if df is None or len(df) < 100:
                        continue

                    # 规则预测
                    predictor = TrendPredictor(df)
                    prediction = predictor.get_comprehensive_prediction()

                    # 获取支撑阻力位
                    latest = df.iloc[-1]
                    support_levels = latest.get('support_levels', [])
                    resistance_levels = latest.get('resistance_levels', [])

                    # ML 预测融合
                    ml_prediction = self._get_ml_prediction(
                        symbol, interval, prediction,
                        {'indicators': {
                            'RSI': latest.get('RSI'),
                            'MACD_Hist': latest.get('MACD_Hist'),
                        }}
                    )
                    if ml_prediction:
                        prediction = self._fuse_predictions(prediction, ml_prediction)

                    # 交易建议
                    trading_advice = predictor.generate_trading_advice(
                        support_levels=support_levels,
                        resistance_levels=resistance_levels
                    )

                    results[symbol][interval] = {
                        'current_price': price_info.get('price') or latest['close'],
                        'stats_24h': price_info.get('stats_24h'),
                        'open_time': latest['open_time'],
                        'open': latest['open'],
                        'high': latest['high'],
                        'low': latest['low'],
                        'close': latest['close'],
                        'volume': latest['volume'],
                        'indicators': {
                            'MA7': latest.get('MA7'),
                            'MA25': latest.get('MA25'),
                            'MA99': latest.get('MA99'),
                            'RSI': latest.get('RSI'),
                            'MACD': latest.get('MACD'),
                            'MACD_Signal': latest.get('MACD_Signal'),
                            'MACD_Hist': latest.get('MACD_Hist'),
                            'BOLL_Upper': latest.get('BOLL_Upper'),
                            'BOLL_Middle': latest.get('BOLL_Middle'),
                            'BOLL_Lower': latest.get('BOLL_Lower'),
                            'ATR': latest.get('ATR'),
                        },
                        'prediction': prediction,
                        'trading_advice': trading_advice,
                        'pattern': latest.get('pattern', ''),
                        'pattern_type': latest.get('pattern_type', ''),
                        'nearest_support': latest.get('nearest_support'),
                        'nearest_resistance': latest.get('nearest_resistance'),
                    }

                # 多周期共振分析
                results[symbol]['resonance'] = self._analyze_resonance(results[symbol])

            # 记录预测并验证历史预测
            self._track_predictions(results, all_prices)

            elapsed = time.time() - start_time
            print(f"[INFO] 数据更新完成，耗时 {elapsed:.2f}s")

            # 更新缓存
            with self._lock:
                self._analysis_cache = results
                self._analysis_time = time.time()

            return results

        finally:
            self._updating = False

    def _analyze_resonance(self, symbol_data: Dict) -> Dict:
        """分析多周期共振"""
        directions = {}
        scores = {}
        volume_weights = {}

        for interval, data in symbol_data.items():
            if interval == 'resonance' or data is None:
                continue
            pred = data.get('prediction', {})
            directions[interval] = pred.get('overall_direction', '观望')
            scores[interval] = pred.get('score', 0)
            volume_weights[interval] = pred.get('volume_weight', 1.0)

        if not directions:
            return {'resonance_level': '无数据', 'signal': '观望', 'confidence': '无'}

        bullish_intervals = []
        bearish_intervals = []

        for interval, direction in directions.items():
            if '涨' in direction or '多' in direction:
                bullish_intervals.append(interval)
            elif '跌' in direction or '空' in direction:
                bearish_intervals.append(interval)

        total_intervals = len(directions)
        bullish_count = len(bullish_intervals)
        bearish_count = len(bearish_intervals)

        interval_weights = {'5m': 0.5, '15m': 0.8, '30m': 1.0, '1h': 1.5, '4h': 2.0, '1d': 3.0}
        weighted_score = 0
        total_weight = 0

        for interval, score in scores.items():
            weight = interval_weights.get(interval, 1.0)
            vol_weight = volume_weights.get(interval, 1.0)
            weighted_score += score * weight * vol_weight
            total_weight += weight

        avg_weighted_score = weighted_score / total_weight if total_weight > 0 else 0

        if bullish_count == total_intervals:
            resonance_level = "完全共振"
            signal = "强烈看涨"
            confidence = "极高"
        elif bearish_count == total_intervals:
            resonance_level = "完全共振"
            signal = "强烈看跌"
            confidence = "极高"
        elif bullish_count >= total_intervals * 0.75:
            resonance_level = "高度共振"
            signal = "看涨"
            confidence = "高"
        elif bearish_count >= total_intervals * 0.75:
            resonance_level = "高度共振"
            signal = "看跌"
            confidence = "高"
        elif bullish_count >= total_intervals * 0.5:
            resonance_level = "部分共振"
            signal = "偏多"
            confidence = "中"
        elif bearish_count >= total_intervals * 0.5:
            resonance_level = "部分共振"
            signal = "偏空"
            confidence = "中"
        else:
            resonance_level = "无共振"
            signal = "观望"
            confidence = "低"

        return {
            'resonance_level': resonance_level,
            'signal': signal,
            'confidence': confidence,
            'bullish_intervals': bullish_intervals,
            'bearish_intervals': bearish_intervals,
            'weighted_score': round(avg_weighted_score, 2),
            'directions': directions
        }

    def _track_predictions(self, results: Dict, prices: Dict):
        """
        记录预测并验证历史预测

        Args:
            results: 分析结果
            prices: 当前价格信息
        """
        tracker = get_tracker()

        # 获取当前价格字典
        current_prices = {
            symbol: info.get('price')
            for symbol, info in prices.items()
            if info.get('price')
        }

        # 验证历史预测
        tracker.verify_predictions(current_prices)

        # 记录新预测（包含特征快照用于反馈学习）
        for symbol, symbol_data in results.items():
            for interval, data in symbol_data.items():
                if interval == 'resonance' or data is None:
                    continue

                prediction = data.get('prediction', {})
                direction = prediction.get('overall_direction', '')
                confidence = prediction.get('confidence', '')
                score = prediction.get('score', 0)
                price = data.get('current_price', 0)

                if direction and price > 0:
                    # 构建特征快照用于反馈学习
                    features = self._extract_features_for_learning(data, prediction)

                    tracker.record_prediction(
                        symbol=symbol,
                        interval=interval,
                        direction=direction,
                        confidence=confidence,
                        score=score,
                        price=price,
                        features=features
                    )

    def _extract_features_for_learning(self, data: Dict, prediction: Dict) -> Dict:
        """
        提取用于反馈学习的特征快照

        Args:
            data: 周期数据
            prediction: 预测结果

        Returns:
            特征字典
        """
        indicators = data.get('indicators', {})
        trend_strength = prediction.get('trend_strength', {})
        momentum = prediction.get('momentum', {})
        obv_divergence = prediction.get('obv_divergence', {})

        features = {
            # 基础指标
            'rsi': indicators.get('RSI'),
            'macd_hist': indicators.get('MACD_Hist'),

            # 预测相关
            'score': prediction.get('score', 0),
            'raw_score': prediction.get('raw_score', 0),
            'volume_weight': prediction.get('volume_weight', 1.0),
            'linear_change_percent': prediction.get('linear_change_percent', 0),
            'bullish_count': prediction.get('bullish_count', 0),
            'bearish_count': prediction.get('bearish_count', 0),

            # ADX 趋势强度
            'adx': trend_strength.get('adx', 0),
            'adx_strength_score': trend_strength.get('strength_score', 0),
            'di_score': trend_strength.get('di_score', 0),
            'plus_di': trend_strength.get('plus_di', 0),
            'minus_di': trend_strength.get('minus_di', 0),

            # 动量
            'momentum_score': momentum.get('score', 0),

            # OBV
            'obv_score': obv_divergence.get('score', 0),

            # 信号相关
            'signals_rsi_score': prediction.get('signals', {}).get('RSI', {}).get('score', 0),
            'signals_macd_score': prediction.get('signals', {}).get('MACD', {}).get('score', 0),
            'signals_ma_score': prediction.get('signals', {}).get('MA', {}).get('score', 0),
            'signals_boll_score': prediction.get('signals', {}).get('BOLL', {}).get('score', 0),
        }

        # 过滤 None 值
        return {k: v for k, v in features.items() if v is not None}

    def _get_ml_prediction(self, symbol: str, interval: str,
                           rule_prediction: Dict, data: Dict) -> Optional[Dict]:
        """
        获取 ML 模型预测

        Args:
            symbol: 交易对
            interval: 周期
            rule_prediction: 规则预测结果
            data: 包含 indicators 的数据

        Returns:
            ML 预测结果或 None
        """
        try:
            learner = get_online_learner()

            # 构建特征
            features = self._extract_features_for_learning(data, rule_prediction)

            # 获取 ML 预测
            ml_result = learner.predict(symbol, interval, features)
            return ml_result
        except Exception as e:
            print(f"[WARN] ML预测失败 {symbol} {interval}: {e}")
            return None

    def _fuse_predictions(self, rule_pred: Dict, ml_pred: Dict) -> Dict:
        """
        融合规则预测与 ML 预测

        权重：规则 60%, ML 40%

        Args:
            rule_pred: 规则预测结果
            ml_pred: ML 预测结果

        Returns:
            融合后的预测结果
        """
        RULE_WEIGHT = 0.6
        ML_WEIGHT = 0.4

        # 从概率中获取 ML 方向分数
        probs = ml_pred.get('probabilities', {})
        bullish_prob = probs.get('bullish', 0.33)
        bearish_prob = probs.get('bearish', 0.33)
        neutral_prob = probs.get('neutral', 0.34)

        # ML 方向分数: bullish_prob - bearish_prob（范围 -1 到 1）
        ml_score = bullish_prob - bearish_prob

        # 确定 ML 方向（用于显示）
        if bullish_prob > bearish_prob and bullish_prob > neutral_prob:
            ml_direction = 'bullish'
        elif bearish_prob > bullish_prob and bearish_prob > neutral_prob:
            ml_direction = 'bearish'
        else:
            ml_direction = 'neutral'

        # ML 置信度调整
        ml_confidence = ml_pred.get('confidence', 0.5)
        ml_score *= ml_confidence

        # 规则预测分数
        rule_score = rule_pred.get('score', 0)

        # 融合分数
        fused_score = rule_score * RULE_WEIGHT + ml_score * ML_WEIGHT

        # 更新预测结果
        result = rule_pred.copy()
        result['score'] = round(fused_score, 2)
        result['raw_rule_score'] = rule_score
        result['ml_prediction'] = {
            'direction': ml_direction,
            'confidence': round(ml_confidence, 2),
            'probabilities': ml_pred.get('probabilities', {}),
            'model_version': ml_pred.get('model_key'),
            'contribution': round(ml_score * ML_WEIGHT, 2)
        }

        # 重新计算方向和置信度
        if fused_score > 1.5:
            result['overall_direction'] = '强烈看涨'
            result['confidence'] = '极高'
        elif fused_score > 0.8:
            result['overall_direction'] = '看涨'
            result['confidence'] = '高'
        elif fused_score > 0.3:
            result['overall_direction'] = '偏多'
            result['confidence'] = '中'
        elif fused_score < -1.5:
            result['overall_direction'] = '强烈看跌'
            result['confidence'] = '极高'
        elif fused_score < -0.8:
            result['overall_direction'] = '看跌'
            result['confidence'] = '高'
        elif fused_score < -0.3:
            result['overall_direction'] = '偏空'
            result['confidence'] = '中'
        else:
            result['overall_direction'] = '震荡观望'
            result['confidence'] = '低'

        return result

    def start_background_update(self, interval: float = 30.0):
        """
        启动后台更新线程

        Args:
            interval: 更新间隔（秒）
        """
        if self._background_thread and self._background_thread.is_alive():
            return

        self._stop_background = False

        def background_worker():
            print("[INFO] 后台数据更新线程启动")
            while not self._stop_background:
                try:
                    self.get_full_analysis(force_refresh=True)
                except Exception as e:
                    print(f"[ERROR] 后台更新失败: {e}")

                # 等待下次更新
                for _ in range(int(interval)):
                    if self._stop_background:
                        break
                    time.sleep(1)

            print("[INFO] 后台数据更新线程停止")

        self._background_thread = threading.Thread(target=background_worker, daemon=True)
        self._background_thread.start()

    def stop_background_update(self):
        """停止后台更新"""
        self._stop_background = True
        if self._background_thread:
            self._background_thread.join(timeout=5)

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'cached_items': len(self._cache),
                'analysis_age': time.time() - self._analysis_time if self._analysis_time else None,
                'is_updating': self._updating,
                'background_running': self._background_thread.is_alive() if self._background_thread else False
            }


# 全局缓存实例
_cache_instance: Optional[DataCache] = None


def get_cache() -> DataCache:
    """获取全局缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance
