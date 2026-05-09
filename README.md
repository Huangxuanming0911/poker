# 德州扑克 CLI 小程序

这是一个基于 Python 的德州扑克逻辑与命令行界面实现，支持 2 名玩家的 PVP 对战。

## 特性

- 52 张牌、洗牌、发牌
- 轮流下注：过牌、跟注、加注、弃牌、全下
- 翻牌圈、转牌圈、河牌圈完整流程
- 手牌比较、胜负判定、分池结算

## 运行方式

建议使用 Python 3.8 及以上版本。

### 1. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 3. 运行程序

```bash
python cli.py
```
