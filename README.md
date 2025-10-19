# 股票分析系统 (Stock Analysis System)

## 简介

基于 https://github.com/DR-lin-eng/stock-scanner 二次修改，感谢原作者  

## 功能变更

1. 增加html页面，支持浏览器在线使用  
2. 增加港股、美股支持  
3. 完善Dockerfile、GitHub Actions 支持docker一键部署使用  
4. 支持x86_64 和 ARM64架构镜像  
5. 支持流式输出，支持前端传入Key(仅作为本地用户使用，日志等内容不会输出) 感谢@Cassianvale  
6. 重构为Vue3+Vite+TS+Naive UI，支持响应式布局  
7. 支持GitHub Actions 一键部署  
8. 支持Nginx反向代理，可通过80/443端口访问

## Docker镜像一键部署

选一种方式即可：

- 方式A：快速启动（单容器，无 Nginx）

```shell
# 拉取最新版本
docker pull qinyz/stock_scanner:latest

# 确保本地目录存在（用于持久化日志和数据）
mkdir -p logs data

# 启动主应用容器（无需自建网络）
docker run -d \
  --name stock-scanner-app \
  -p 8888:8888 \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/data:/app/data" \
  -e API_KEY="你的API密钥" \
  -e API_URL="你的API地址" \
  -e API_MODEL="你的API模型" \
  -e API_TIMEOUT="60" \
  -e LOGIN_PASSWORD="你的登录密码" \
  -e ANNOUNCEMENT_TEXT="你的公告内容" \
  --restart unless-stopped \
  qinyz/stock_scanner:latest
```

- 方式B：使用 docker-compose（推荐，自动拉取镜像）

```shell
# 克隆仓库以获取 compose 文件和 Nginx 配置
git clone https://github.com/qinyz/stock_scanner.git
cd stock_scanner

# 创建.env并设置必要变量（参考下文说明）
cp .env.example .env
# 然后编辑 .env 填写 API_KEY / API_URL / API_MODEL 等

# 启动（从 Docker Hub 拉取镜像）
docker-compose -f docker-compose.simple.yml up -d
```

- 方式C：手动运行 Nginx 反向代理（可选）

```shell
# 创建自定义网络（App 和 Nginx 通过同一网络通信）
docker network create stock-scanner-network

# 启动主应用容器
docker run -d \
  --name stock-scanner-app \
  --network stock-scanner-network \
  -p 8888:8888 \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/data:/app/data" \
  -e API_KEY="你的API密钥" \
  -e API_URL="你的API地址" \
  -e API_MODEL="你的API模型" \
  -e API_TIMEOUT="60" \
  -e LOGIN_PASSWORD="你的登录密码" \
  -e ANNOUNCEMENT_TEXT="你的公告内容" \
  --restart unless-stopped \
  qinyz/stock_scanner:latest

# 准备 Nginx 配置和目录（使用仓库中的 nginx/nginx.conf）
mkdir -p nginx/logs nginx/ssl

# 启动 Nginx 容器（将本地配置和证书目录挂载到容器）
docker run -d \
  --name stock-scanner-nginx \
  --network stock-scanner-network \
  -p 80:80 \
  -p 443:443 \
  -v "$(pwd)/nginx/nginx.conf:/etc/nginx/conf.d/default.conf" \
  -v "$(pwd)/nginx/logs:/var/log/nginx" \
  -v "$(pwd)/nginx/ssl:/etc/nginx/ssl" \
  --restart unless-stopped \
  nginx:stable-alpine
```

API_URL 处理逻辑（与 Cherry Studio 保持一致）：
- 当 API_URL 以 / 结尾时直接追加 chat/completions，保留原有版本号：
  - 输入: https://ark.cn-beijing.volces.com/api/v3/
  - 输出: https://ark.cn-beijing.volces.com/api/v3/chat/completions
- 当 API_URL 以 # 结尾时强制使用当前链接：
  - 输入: https://ark.cn-beijing.volces.com/api/v3/chat/completions#
  - 输出: https://ark.cn-beijing.volces.com/api/v3/chat/completions
- 当 API_URL 不以 / 结尾时使用默认版本号 v1：
  - 输入: https://ark.cn-beijing.volces.com/api
  - 输出: https://ark.cn-beijing.volces.com/api/v1/chat/completions

默认 8888 端口，部署完成后访问 http://你的域名或 IP:8888 即可使用  

## 使用Nginx反向代理

项目已集成Nginx服务，可以通过80端口(HTTP)和443端口(HTTPS)访问应用  
使用docker-compose启动：  

```shell
# 克隆仓库
git clone https://github.com/qinyz/stock_scanner.git
cd stock_scanner

# 创建.env文件并填写必要的环境变量
cat > .env << EOL
API_KEY=你的API密钥
API_URL=你的API地址
API_MODEL=你的API模型
API_TIMEOUT=超时时间(默认60秒)
LOGIN_PASSWORD=登录密码(可选)
ANNOUNCEMENT_TEXT=公告文本
EOL

# 创建SSL证书目录
mkdir -p nginx/ssl

# 生成自签名SSL证书（仅用于测试环境）
openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem \
  -subj "/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# 启动服务（包含 Nginx）
docker-compose -f docker-compose.simple.yml up -d
```

### 使用自己的SSL证书

如果您有自己的SSL证书，可以替换自签名证书：

1. 将您的证书文件放在 `nginx/ssl/` 目录下
2. 确保证书文件命名为 `fullchain.pem`，私钥文件命名为 `privkey.pem`
3. 重启服务: `docker-compose -f docker-compose.simple.yml restart nginx`

相关参考：[免费泛域名 SSL 证书申请及自动续期（使用 1Panel 面板）](https://bronya-zaychik.cn/archives/GenSSL.html)

## Github Actions 部署

| 环境变量 | 说明 |
| --- | --- |
| DOCKERHUB_USERNAME | Docker Hub用户名 |
| DOCKERHUB_TOKEN | Docker Hub访问令牌 |
| SERVER_HOST | 部署服务器地址 |
| SERVER_USERNAME | 服务器用户名 |
| SSH_PRIVATE_KEY | SSH私钥 |
| DEPLOY_PATH | 部署路径 |
| SLACK_WEBHOOK | Slack通知Webhook（可选） |


## 注意事项 (Notes)
- 股票分析仅供参考，不构成投资建议
- 使用前请确保网络连接正常
- 建议在实盘前充分测试

## 贡献 (Contributing)
欢迎提交 issues 和 pull requests！

## 许可证 (License)
[待添加具体许可证信息]

## 免责声明 (Disclaimer)
本系统仅用于学习和研究目的，投资有风险，入市需谨慎。

## Quant v2（实验性，特性开关控制）

概览
- v2 在 /api/v2 下提供一组可插拔的量化与推荐能力，均通过特性开关启用/禁用。包含模块：factors、backtest、portfolio、risk、ml、recommendation。
- v1 的接口与 UI 保持不变，完全兼容。v2 为增量特性，不会破坏现有使用方式。

模块与用途
- factors：按需计算经典技术指标因子（SMA/EMA/RSI/MACD/ATR/BBANDS）。
- backtest：异步任务化的简易买入并持有回测，支持进度日志流式输出。
- portfolio：基础组合权重计算（等权、波动率倒数占位实现）。
- risk：滚动波动率、最大回撤、平均收益、Calmar 比率、风险调整分数（在评分流程中复用）。
- ml：基线动量预测示例，可替换为自定义模型。
- recommendation：将标的分数聚合为评级与交易计划（止盈/止损/持有期），支持风险偏好。

向后兼容性
- v1 接口保持不变；v2 接口统一位于 /api/v2，可按环境通过特性开关启用。

功能开关（默认：全部关闭）
- ENABLE_FACTORS=false  因子模块
- ENABLE_BACKTEST=false 回测模块
- ENABLE_PORTFOLIO=false 组合模块
- ENABLE_RISK=false 风险模块
- ENABLE_ML=false 机器学习模块
- ENABLE_RECO=false 推荐模块

启用方式
- .env 文件：将以上变量置为 true（参考 .env.example）
- Shell：export ENABLE_FACTORS=true ENABLE_BACKTEST=true ...
- Docker Compose：在 app 服务的 environment 中添加上述变量

依赖
- Redis（推荐）：用于缓存因子计算等；不可用时自动回退到内存。配置 REDIS_URL 或 REDIS_HOST/REDIS_PORT/REDIS_DB；健康检查：GET /api/v2/health/cache。
- Worker：回测使用进程内线程池异步执行，无需额外 Celery/RQ。若需更高吞吐，可部署独立任务系统；当前版本提供流式进度输出。
- 可选 MLflow：如需对接自有模型注册/追踪，可在外部集成 MLflow；基础功能不要求。

Docker Compose（本地 + Redis）
```
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  app:
    image: qinyz/stock_scanner:latest
    ports:
      - "8888:8888"
    environment:
      - API_KEY=${API_KEY}
      - API_URL=${API_URL}
      - API_MODEL=${API_MODEL}
      - API_TIMEOUT=${API_TIMEOUT}
      - LOGIN_PASSWORD=${LOGIN_PASSWORD}
      - ANNOUNCEMENT_TEXT=${ANNOUNCEMENT_TEXT}
      - ENABLE_FACTORS=true
      - ENABLE_BACKTEST=true
      - ENABLE_PORTFOLIO=true
      - ENABLE_RISK=true
      - ENABLE_ML=true
      - ENABLE_RECO=true
      - ENABLE_CACHE=true
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
```

资源建议
- 小型开发机：应用 2 vCPU / 2 GB RAM；Redis 256–512 MB。回测为 CPU 密集型；可通过提升线程或多实例扩容以提高吞吐。
- 磁盘：日志与（可选）快照约 200 MB 起；启用 DuckDB 快照后 data/ 目录体量会增加。
- 网络：Akshare 拉取 I/O 密集；在受限网络可考虑本地镜像。

API 使用示例（均位于 /api/v2）
- 列出可用因子
```
curl -s http://localhost:8888/api/v2/factors
```

- 计算因子
```
curl -s -X POST http://localhost:8888/api/v2/factors/compute \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["600519", "AAPL"],
    "market": "A",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "factors": [
      {"id": "sma", "params": {"period": 20}, "output": "last"},
      {"id": "rsi", "params": {"period": 14}, "output": "last"},
      {"id": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}, "output": "series"}
    ]
  }'
```

- 运行回测（异步任务）
```
# 1) 提交任务
curl -s -X POST http://localhost:8888/api/v2/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"],
    "market": "US",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "params": {"initial_cash": 100000, "fee_bps": 1, "slippage_bps": 1}
  }'
# => {"job_id":"<id>"}

# 2) 轮询任务状态/结果
curl -s http://localhost:8888/api/v2/backtest/<job_id>

# 3) 流式获取日志/进度（按行 JSON）
curl -s -N http://localhost:8888/api/v2/backtest/<job_id>/stream
```

- 组合优化
```
curl -s -X POST http://localhost:8888/api/v2/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL","MSFT","GOOG"], "method": "equal"}'
```

- 机器学习预测（基线动量）
```
curl -s -X POST http://localhost:8888/api/v2/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL","MSFT"],
    "market": "US",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31"
  }'
```

- 交易信号推荐
```
curl -s -X POST http://localhost:8888/api/v2/signals/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL","MSFT"],
    "risk_appetite": "balanced",
    "scores": {"AAPL": 0.8, "MSFT": 0.2}
  }'
```

数据迁移
- 默认使用 DuckDB/CSV（可选）保存快照，无需数据库迁移。
- 若引入 SQL 数据库，遵循“仅增不破坏”迁移策略。建议：
  - 生成迁移：alembic revision -m "add <table/column>"
  - 应用迁移：alembic upgrade head
  - 只前滚；发布期间不做删除/重命名
  - 先在 Staging 执行迁移并做因子计算与回测冒烟，再发布到生产

安全与限制
- JWT：当配置 LOGIN_PASSWORD 时，v1 接口需先通过 POST /api/login 获取令牌并在请求头携带 Authorization: Bearer <token>。v2 目前默认不强制 JWT；如需保护，请在网络策略/Ingress/反向代理层做访问控制。
- 频率限制：未内置限流；可在 Nginx/Ingress 或 API 网关配置。
- 请求与负载：为保证延迟，建议单次 symbols <= 100；小规格机器回测建议每任务 <= 20 个标的。
- ML 防泄漏：基线 ML 不访问外部服务。接入自定义模型时，请避免在日志/返回中泄露私有数据，做好脱敏与校验。

可观测性
- /metrics 暴露 Prometheus 指标（安装 prometheus_client 时有效）：
  - api_latency_seconds 直方图（标签 path, method, status）
  - cache_hits_total 与 cache_misses_total 计数器
  - backtest_job_duration_seconds 直方图
  - process_memory_rss_bytes Gauge
- Prometheus 抓取配置示例
```
scrape_configs:
  - job_name: stock-scanner
    static_configs:
      - targets: ['localhost:8888']
    metrics_path: /metrics
```

- 仪表盘建议：
  - API 延迟分布（按 path/method 维度）
  - 缓存命中率与未命中数（hits/misses）
  - 回测任务时长分布（成功/失败）

快速开始（本地开发）
- 安装
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
- 启动 Redis（可选，推荐）
```
docker run -d --name redis -p 6379:6379 redis:7-alpine
```
- 启用 v2 并启动 API
```
export ENABLE_FACTORS=true ENABLE_BACKTEST=true ENABLE_PORTFOLIO=true ENABLE_RISK=true ENABLE_ML=true ENABLE_RECO=true
export ENABLE_CACHE=true REDIS_HOST=localhost REDIS_PORT=6379
uvicorn web_server:app --host 0.0.0.0 --port 8888 --reload
```
- 端到端示例
```
curl -s -X POST http://localhost:8888/api/v2/factors/compute \
  -H "Content-Type: application/json" \
  -d '{"symbols":["600519"],"market":"A","factors":[{"id":"sma","params":{"period":20}}]}'
```

OpenAPI
- Swagger UI: http://localhost:8888/docs（查看标签“v2”）；ReDoc: http://localhost:8888/redoc
- v1 保持支持不变；v2 为 /api/v2 下的增量接口

可观测性说明
- 未安装 prometheus_client 时，/metrics 返回空内容，指标为 no-op。

缓存说明
- utils.cache.Cache 在存在 Redis 时自动使用 Redis，否则回退到内存；内置函数装饰器、健康检查与命中/未命中指标。
