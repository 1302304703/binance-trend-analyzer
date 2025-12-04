# Binance Trend Analyzer

[![Version](https://img.shields.io/badge/version-1.6.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> 币安交易对趋势分析与预测系统，支持多周期共振分析、智能交易建议、K线形态识别、机器学习预测

**[查看更新日志 (CHANGELOG.md)](CHANGELOG.md)**

---

## 最新更新 v1.6.0

### 性能大幅提升
| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| API响应时间 | 10-15秒 | **0.02秒** | 500倍 |

- 并行数据获取（10线程）
- 内存缓存 + 后台预加载
- 页面秒级响应

### 智能交易建议 (v1.5.0)
- 自动生成做多/做空/观望建议
- 止盈止损位自动计算（基于ATR+支撑阻力）
- 合约杠杆倍数建议（根据波动率动态调整）
- 仓位建议 + 风险等级评估

---

## 核心功能

### 趋势分析
- **多周期分析**: 5分钟、30分钟、1小时、4小时、日线
- **技术指标**: MA、EMA、RSI、Stochastic RSI、MACD、布林带、ATR、ADX、OBV、动量
- **多周期共振**: 判断各周期方向一致性，综合置信度评级

### 智能交易建议
- **开仓建议**: 基于多指标共振 + ADX趋势强度
- **止盈止损**: ATR动态计算 + 支撑阻力位优化
- **杠杆建议**: 综合波动率、置信度、趋势强度
- **仓位管理**: 根据置信度分级建议仓位比例

### 形态识别
- **反转形态**: 锤子线、倒锤子线、射击之星、吞没形态、早晨/黄昏之星等
- **趋势形态**: 三只白兵、三只黑鸦、大阳线、大阴线
- **悬浮提示**: 18种形态详细说明

### 机器学习
- **XGBoost 模型**: 30+特征的梯度提升分类
- **LSTM 模型**: 深度学习时序预测（可选）
- **三分类预测**: 上涨/横盘/下跌

### 可视化
- **趋势面板**: 实时刷新的多周期分析面板
- **K线图表**: TradingView风格专业K线图

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/1302304703/binance-trend-analyzer.git
cd binance-trend-analyzer
```

### 2. 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行

**Web界面模式（推荐）:**

```bash
python3 src/app.py
```

- 趋势面板: http://127.0.0.1:5000
- K线图表: http://127.0.0.1:5000/chart

**命令行模式:**

```bash
python3 src/main.py
```

**运行回测:**

```bash
python3 src/backtest.py
```

**训练机器学习模型:**

```bash
python3 src/ml_predictor.py
```

---

## 交易建议说明

### 建议类型

| 建议 | 含义 | 置信度要求 |
|------|------|-----------|
| 建议做多 | 强烈看涨信号 | 高/极高 |
| 可考虑做多 | 中等看涨信号 | 中高 |
| 观望 | 信号不明确 | 低/中 |
| 建议做空 | 强烈看跌信号 | 高/极高 |

### 杠杆建议逻辑

| 置信度 | 最大杠杆 | 说明 |
|--------|----------|------|
| 极高 | 20x | 强趋势 + 低波动 |
| 高 | 15x | 趋势明确 |
| 中高 | 10x | 谨慎操作 |
| 其他 | 5x | 轻仓试探 |

### 止盈止损

- **止损**: 1.5倍ATR 或 最近支撑/阻力位
- **止盈1 (TP1)**: 1.5倍风险收益比，平仓30%
- **止盈2 (TP2)**: 2.5倍风险收益比，平仓40%
- **止盈3 (TP3)**: 4倍风险收益比，平仓30%

---

## 项目结构

```
binance-trend-analyzer/
├── requirements.txt      # 依赖
├── README.md             # 项目说明
├── CHANGELOG.md          # 更新日志
├── models/               # 保存的ML模型
└── src/
    ├── config.py         # 配置
    ├── collector.py      # 数据采集
    ├── data_cache.py     # 数据缓存（性能优化）
    ├── indicators.py     # 技术指标 + K线形态 + 支撑阻力
    ├── predictor.py      # 规则预测 + 交易建议
    ├── ml_predictor.py   # 机器学习预测（XGBoost/LSTM）
    ├── analyzer.py       # 趋势分析
    ├── backtest.py       # 回测系统
    ├── main.py           # 命令行入口
    ├── app.py            # Web服务入口
    └── templates/
        ├── index.html    # 趋势面板页面
        └── chart.html    # K线图表页面
```

---

## API 接口

| 接口 | 说明 |
|------|------|
| `/api/analysis` | 获取完整分析数据（含交易建议） |
| `/api/klines` | 获取K线数据（用于图表） |
| `/api/cache-stats` | 获取缓存状态 |

---

## 配置说明

编辑 `src/config.py` 自定义配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| SYMBOLS | BTCUSDT, ETHUSDT | 监控的交易对 |
| INTERVALS | 5m, 30m, 1h, 4h, 1d | K线周期 |
| REFRESH_INTERVAL | 60 | 前端刷新间隔(秒) |

---

## 回测结果示例

```
=== BTCUSDT 1h 回测结果 ===
测试样本: 100
预测准确率: 70.7%
交易胜率: 70.2%
累计收益: 30.41%
```

---

## 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。加密货币市场波动剧烈，投资有风险，入市需谨慎。

## License

MIT
