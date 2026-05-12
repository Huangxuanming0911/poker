# 德州扑克 Texas Hold'em

一个 Python 实现的德州扑克游戏，包含两种玩法：**命令行**和**网页 UI**（中英双语）。
AI 支持范围建模 + Fold Equity 决策 + 自我对战训练。

## 特性

- **完整德扑流程**：双人单挑，含翻牌前/翻牌/转牌/河牌、盲注、过牌/跟注/加注/全下/弃牌、分池结算
- **两个 AI 等级**：
  - `BasicPokerAI`：手工启发式（轻量）
  - `AdvancedPokerAI`：169-起手牌查表 + Monte Carlo 胜率 + 对手范围建模 + Fold Equity + 对手画像（VPIP/PFR/AF/Fold-to-Cbet/WTSD 贝叶斯平滑）
- **网页 UI**：
  - 中/英文双语切换（设置面板）
  - 卡牌双模式：CSS 渲染 / Byron Knoll SVG 牌组
  - 单人 vs AI（PVE）+ 同机双人对战（PVP，含挡屏切换）
  - All-in 逐张翻牌动画（点击屏幕可跳过）
- **离线训练**：(1+λ)-ES 演化策略，多进程并行 self-play

## 快速开始

需要 **Python 3.8+**。

### 1. 创建虚拟环境 + 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### 2A. 命令行版

```bash
python cli.py
```

按提示选择 PVE/PVP 模式、输入名字、选 AI 等级。
动作输入支持编号或别名（`1`/`fold`/`f`、`2`/`call`/`ca`、`3`/`r2`/`2x`、`4`/`r3`/`3x`、`5`/`allin`/`a`）。

> ⚠️ 注意：`c` 是 **check**（过牌），跟注请用 `ca` 或编号 `2`。

### 2B. 网页 UI 版（推荐）

```bash
# 第一次运行：下载公共领域 SVG 卡牌（一次性，约 3MB，否则 SVG 模式不可用）
python scripts/download_cards.py

# 启动服务器
python server.py
```

打开浏览器访问 **<http://localhost:5000>**。

- 选模式：vs AI / 双人对战
- 顶栏 ⚙ 切换卡牌样式（CSS / SVG）和语言（中文 / English）
- PVP 模式下每次玩家切换会显示挡屏，避免对手偷看你的牌
- 一手结束后底部抽屉显示摊牌 + 胜负，点击"下一手"继续

> 💡 第一次启动会自动构建 169-起手牌胜率查表（约 10 秒，结果缓存到 `equity_preflop_table.json`）。

### 2C. 评测 / 训练 AI（可选）

快速评测当前 AI vs Basic AI：
```bash
python train.py --eval-only --hands 1000 --seeds 5
```

后台训练参数（20 代演化，4 进程并行，约 2-4 小时）：
```bash
python train.py --generations 20 --lambda 6 --hands 1500 --workers 4
```

训练结果保存到 `data/ai_params.json`，下次启动自动加载。

## 项目结构

```
poker/
├── engine.py              # 核心引擎：发牌、betting_round、手牌评估、分池
├── ai.py                  # BasicPokerAI（启发式基线）
├── advanced_ai.py         # AdvancedPokerAI（基于范围 + fold equity）
├── ai_params.py           # 13 维可训练参数 + JSON 持久化
├── equity.py              # 胜率引擎：preflop 查表 + MC + river 精确枚举
├── range.py               # 对手范围模型 + 加权胜率 + fold equity 估算
├── opponent_model.py      # VPIP/PFR/AF 统计 + 贝叶斯平滑
├── persistence.py         # HandLog 序列化、jsonl 历史、画像 JSON
├── train.py               # (1+λ)-ES self-play 训练 + 评测
├── cli.py                 # 命令行交互
├── server.py              # Flask 网页后端
├── scripts/
│   └── download_cards.py  # 下载 Byron Knoll SVG 卡牌
├── static/                # 网页前端
│   ├── index.html
│   ├── css/               # style/table/card 三块样式
│   ├── js/                # app/api/card/i18n 四个模块
│   ├── i18n/              # zh.json + en.json
│   └── img/cards/         # SVG 卡牌（gitignore，跑下载脚本生成）
├── data/                  # 运行时生成（gitignore）：ai_params / 画像 / 手牌日志
└── tests/                 # （计划中）
```

## 开发笔记

- 引擎 `engine.Game.play_hand(action_provider)` 是同步阻塞的；网页后端 `server.py` 用每会话一个 daemon 线程 + 两个 `queue.Queue` 把它包装成 HTTP 请求/响应模型
- `DecisionContext`（[engine.py](engine.py)）暴露给 AI 的字段包括 `available_actions`、`action_targets`、`actions_this_hand`（动作序列）、`button_index` —— AI 可独立做范围回放，无需额外引擎钩子
- `AIParams.clip_ranges()` 定义每个参数的合法区间；`train.py` 的演化策略在这个区间内做高斯扰动 + clip

## 开源资源致谢

- **SVG 卡牌牌组**：[Byron Knoll 公共领域牌组](https://github.com/notpeter/Vector-Playing-Cards)
- **UI 设计灵感**：[Poker Table by goodo73](https://codepen.io/goodo73/pen/zYQGWz)、[Blackjack felt by StarDrop9](https://codepen.io/StarDrop9/pen/XxBrOG)
- **后端架构参考**：[jarczano/Texas-Holdem-Poker-Web-App](https://github.com/jarczano/Texas-Holdem-Poker-Web-App)

## License

代码原创部分采用 MIT 协议。引入的 SVG 卡牌（运行时下载，不在仓库中）为公共领域作品（Byron Knoll）。
