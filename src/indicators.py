"""
技术指标计算模块
@author Reln Ding
"""

import pandas as pd
import numpy as np
from config import MA_PERIODS, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BOLL_PERIOD, BOLL_STD


def calculate_ma(df: pd.DataFrame, periods: list = MA_PERIODS) -> pd.DataFrame:
    """
    计算移动平均线

    Args:
        df: K线数据
        periods: MA周期列表

    Returns:
        添加MA列后的DataFrame
    """
    for period in periods:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()
    return df


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """
    计算指数移动平均线

    Args:
        df: K线数据
        period: EMA周期

    Returns:
        EMA序列
    """
    return df['close'].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    计算RSI指标

    Args:
        df: K线数据
        period: RSI周期

    Returns:
        添加RSI列后的DataFrame
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW,
                   signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """
    计算MACD指标

    Args:
        df: K线数据
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期

    Returns:
        添加MACD相关列后的DataFrame
    """
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def calculate_bollinger(df: pd.DataFrame, period: int = BOLL_PERIOD,
                        std_dev: int = BOLL_STD) -> pd.DataFrame:
    """
    计算布林带

    Args:
        df: K线数据
        period: 布林带周期
        std_dev: 标准差倍数

    Returns:
        添加布林带列后的DataFrame
    """
    df['BOLL_Middle'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['BOLL_Upper'] = df['BOLL_Middle'] + (rolling_std * std_dev)
    df['BOLL_Lower'] = df['BOLL_Middle'] - (rolling_std * std_dev)
    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算ATR（平均真实波幅）

    Args:
        df: K线数据
        period: ATR周期

    Returns:
        添加ATR列后的DataFrame
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean()
    return df


def calculate_volume_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算成交量移动平均

    Args:
        df: K线数据
        period: 周期

    Returns:
        添加成交量MA后的DataFrame
    """
    df['Volume_MA'] = df['volume'].rolling(window=period).mean()
    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算ADX（平均趋向指数）- 用于判断趋势强度

    Args:
        df: K线数据
        period: ADX周期

    Returns:
        添加ADX、+DI、-DI列后的DataFrame
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # 计算+DM和-DM
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1

    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

    # 重新计算
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    # 当+DM > -DM时，-DM = 0；反之亦然
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

    # 计算TR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 平滑TR、+DM、-DM
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

    # 计算+DI和-DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # 计算DX和ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(span=period, adjust=False).mean()

    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = adx

    return df


def calculate_stoch_rsi(df: pd.DataFrame, rsi_period: int = 14,
                        stoch_period: int = 14, k_period: int = 3,
                        d_period: int = 3) -> pd.DataFrame:
    """
    计算Stochastic RSI - 比RSI更敏感的超买超卖指标

    Args:
        df: K线数据
        rsi_period: RSI周期
        stoch_period: Stochastic周期
        k_period: K线平滑周期
        d_period: D线平滑周期

    Returns:
        添加StochRSI_K和StochRSI_D后的DataFrame
    """
    # 确保RSI已计算
    if 'RSI' not in df.columns:
        df = calculate_rsi(df, rsi_period)

    rsi = df['RSI']

    # 计算Stochastic RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()

    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

    # K线和D线
    df['StochRSI_K'] = stoch_rsi.rolling(window=k_period).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(window=d_period).mean()

    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算OBV（能量潮）- 量价关系分析

    Args:
        df: K线数据

    Returns:
        添加OBV和OBV_MA后的DataFrame
    """
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])

    df['OBV'] = obv
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    return df


def calculate_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    计算动量指标

    Args:
        df: K线数据
        period: 动量周期

    Returns:
        添加Momentum和ROC后的DataFrame
    """
    # 动量 = 当前价格 - N周期前价格
    df['Momentum'] = df['close'] - df['close'].shift(period)

    # ROC (变化率) = (当前价格 - N周期前价格) / N周期前价格 * 100
    df['ROC'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

    return df


def calculate_ema_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算EMA趋势指标

    Args:
        df: K线数据

    Returns:
        添加EMA指标后的DataFrame
    """
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    return df


def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别K线形态

    Args:
        df: K线数据

    Returns:
        添加形态识别列后的DataFrame
    """
    df = df.copy()

    # 计算K线实体和影线
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']

    # 平均实体大小（用于判断大小阳线/阴线）
    avg_body = df['body_abs'].rolling(window=20).mean()

    # 初始化形态列
    df['pattern'] = ''
    df['pattern_type'] = ''  # bullish, bearish, neutral
    df['pattern_strength'] = 0  # 形态强度 1-3

    for i in range(2, len(df)):
        patterns = []
        pattern_type = 'neutral'
        strength = 0

        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        prev2 = df.iloc[i - 2]
        avg = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else curr['body_abs']

        # 避免除零
        if curr['total_range'] == 0:
            continue

        body_ratio = curr['body_abs'] / curr['total_range']
        upper_ratio = curr['upper_shadow'] / curr['total_range']
        lower_ratio = curr['lower_shadow'] / curr['total_range']

        # ============ 反转形态 ============

        # 1. 锤子线 (Hammer) - 看涨反转
        if (lower_ratio > 0.6 and body_ratio < 0.3 and upper_ratio < 0.1 and
                curr['close'] < prev['close']):
            patterns.append('锤子线')
            pattern_type = 'bullish'
            strength = 2

        # 2. 倒锤子线 (Inverted Hammer) - 看涨反转
        if (upper_ratio > 0.6 and body_ratio < 0.3 and lower_ratio < 0.1 and
                curr['close'] < prev['close']):
            patterns.append('倒锤子线')
            pattern_type = 'bullish'
            strength = 2

        # 3. 上吊线 (Hanging Man) - 看跌反转
        if (lower_ratio > 0.6 and body_ratio < 0.3 and upper_ratio < 0.1 and
                curr['close'] > prev['close']):
            patterns.append('上吊线')
            pattern_type = 'bearish'
            strength = 2

        # 4. 射击之星 (Shooting Star) - 看跌反转
        if (upper_ratio > 0.6 and body_ratio < 0.3 and lower_ratio < 0.1 and
                curr['close'] > prev['close']):
            patterns.append('射击之星')
            pattern_type = 'bearish'
            strength = 2

        # 5. 十字星 (Doji)
        if body_ratio < 0.1:
            if upper_ratio > 0.4 and lower_ratio > 0.4:
                patterns.append('长腿十字星')
                strength = 2
            elif upper_ratio < 0.1 and lower_ratio > 0.6:
                patterns.append('蜻蜓十字星')
                pattern_type = 'bullish'
                strength = 2
            elif lower_ratio < 0.1 and upper_ratio > 0.6:
                patterns.append('墓碑十字星')
                pattern_type = 'bearish'
                strength = 2
            else:
                patterns.append('十字星')
                strength = 1

        # 6. 看涨吞没 (Bullish Engulfing)
        if (prev['body'] < 0 and curr['body'] > 0 and
                curr['open'] < prev['close'] and curr['close'] > prev['open'] and
                curr['body_abs'] > prev['body_abs']):
            patterns.append('看涨吞没')
            pattern_type = 'bullish'
            strength = 3

        # 7. 看跌吞没 (Bearish Engulfing)
        if (prev['body'] > 0 and curr['body'] < 0 and
                curr['open'] > prev['close'] and curr['close'] < prev['open'] and
                curr['body_abs'] > prev['body_abs']):
            patterns.append('看跌吞没')
            pattern_type = 'bearish'
            strength = 3

        # 8. 乌云盖顶 (Dark Cloud Cover)
        if (prev['body'] > 0 and curr['body'] < 0 and
                curr['open'] > prev['high'] and
                curr['close'] < (prev['open'] + prev['close']) / 2 and
                curr['close'] > prev['open']):
            patterns.append('乌云盖顶')
            pattern_type = 'bearish'
            strength = 2

        # 9. 刺透形态 (Piercing Pattern)
        if (prev['body'] < 0 and curr['body'] > 0 and
                curr['open'] < prev['low'] and
                curr['close'] > (prev['open'] + prev['close']) / 2 and
                curr['close'] < prev['open']):
            patterns.append('刺透形态')
            pattern_type = 'bullish'
            strength = 2

        # 10. 早晨之星 (Morning Star) - 三根K线
        if (prev2['body'] < 0 and prev2['body_abs'] > avg * 0.5 and
                prev['body_abs'] < avg * 0.3 and
                curr['body'] > 0 and curr['body_abs'] > avg * 0.5 and
                curr['close'] > (prev2['open'] + prev2['close']) / 2):
            patterns.append('早晨之星')
            pattern_type = 'bullish'
            strength = 3

        # 11. 黄昏之星 (Evening Star) - 三根K线
        if (prev2['body'] > 0 and prev2['body_abs'] > avg * 0.5 and
                prev['body_abs'] < avg * 0.3 and
                curr['body'] < 0 and curr['body_abs'] > avg * 0.5 and
                curr['close'] < (prev2['open'] + prev2['close']) / 2):
            patterns.append('黄昏之星')
            pattern_type = 'bearish'
            strength = 3

        # 12. 三只白兵 (Three White Soldiers)
        if (prev2['body'] > 0 and prev['body'] > 0 and curr['body'] > 0 and
                prev['close'] > prev2['close'] and curr['close'] > prev['close'] and
                prev['body_abs'] > avg * 0.5 and curr['body_abs'] > avg * 0.5):
            patterns.append('三只白兵')
            pattern_type = 'bullish'
            strength = 3

        # 13. 三只黑鸦 (Three Black Crows)
        if (prev2['body'] < 0 and prev['body'] < 0 and curr['body'] < 0 and
                prev['close'] < prev2['close'] and curr['close'] < prev['close'] and
                prev['body_abs'] > avg * 0.5 and curr['body_abs'] > avg * 0.5):
            patterns.append('三只黑鸦')
            pattern_type = 'bearish'
            strength = 3

        # ============ 持续形态 ============

        # 14. 大阳线
        if curr['body'] > 0 and curr['body_abs'] > avg * 1.5:
            patterns.append('大阳线')
            pattern_type = 'bullish'
            strength = max(strength, 2)

        # 15. 大阴线
        if curr['body'] < 0 and curr['body_abs'] > avg * 1.5:
            patterns.append('大阴线')
            pattern_type = 'bearish'
            strength = max(strength, 2)

        # 保存识别结果
        if patterns:
            df.iloc[i, df.columns.get_loc('pattern')] = ','.join(patterns)
            df.iloc[i, df.columns.get_loc('pattern_type')] = pattern_type
            df.iloc[i, df.columns.get_loc('pattern_strength')] = strength

    # 清理临时列
    df = df.drop(columns=['body', 'body_abs', 'upper_shadow', 'lower_shadow', 'total_range'])

    return df


def calculate_support_resistance(df: pd.DataFrame, window: int = 20, num_levels: int = 3) -> pd.DataFrame:
    """
    识别支撑位和阻力位

    Args:
        df: K线数据
        window: 用于识别高低点的窗口大小
        num_levels: 返回的支撑/阻力位数量

    Returns:
        添加支撑阻力位后的DataFrame
    """
    df = df.copy()

    # 识别局部高点和低点
    highs = []
    lows = []

    for i in range(window, len(df) - window):
        # 局部最高点
        if df['high'].iloc[i] == df['high'].iloc[i - window:i + window + 1].max():
            highs.append(df['high'].iloc[i])
        # 局部最低点
        if df['low'].iloc[i] == df['low'].iloc[i - window:i + window + 1].min():
            lows.append(df['low'].iloc[i])

    # 聚类相近的价位
    def cluster_levels(levels, threshold_pct=0.5):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] * 100 < threshold_pct:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        clusters.append(np.mean(current_cluster))

        # 按出现频率排序（这里简化为按价格排序）
        return sorted(clusters, reverse=True)[:num_levels * 2]

    resistance_levels = cluster_levels(highs)
    support_levels = cluster_levels(lows)

    current_price = df['close'].iloc[-1]

    # 找出当前价格上方的阻力位和下方的支撑位
    resistances = [r for r in resistance_levels if r > current_price][:num_levels]
    supports = [s for s in support_levels if s < current_price][-num_levels:]

    # 添加到DataFrame
    df['support_levels'] = [supports] * len(df)
    df['resistance_levels'] = [resistances] * len(df)

    # 计算最近支撑和阻力
    df['nearest_support'] = max(supports) if supports else None
    df['nearest_resistance'] = min(resistances) if resistances else None

    # 计算距离支撑/阻力的百分比
    if supports:
        df['distance_to_support'] = (current_price - max(supports)) / current_price * 100
    else:
        df['distance_to_support'] = None

    if resistances:
        df['distance_to_resistance'] = (min(resistances) - current_price) / current_price * 100
    else:
        df['distance_to_resistance'] = None

    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有技术指标

    Args:
        df: K线数据

    Returns:
        添加所有指标后的DataFrame
    """
    df = calculate_ma(df)
    df = calculate_ema_trend(df)
    df = calculate_rsi(df)
    df = calculate_stoch_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger(df)
    df = calculate_atr(df)
    df = calculate_adx(df)
    df = calculate_obv(df)
    df = calculate_momentum(df)
    df = calculate_volume_ma(df)
    df = calculate_candlestick_patterns(df)
    df = calculate_support_resistance(df)
    return df
