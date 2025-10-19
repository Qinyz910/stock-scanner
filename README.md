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

## Quant v2 (Experimental, feature-flagged)

Overview
- v2 introduces modular quantitative research and recommendation capabilities behind feature flags. Modules: factors, backtest, portfolio, risk, ml, recommendation.
- v1 API and UI remain unchanged and fully supported. v2 is additive and non-breaking.

Modules and purpose
- factors: compute classic TA factors (SMA/EMA/RSI/MACD/ATR/BBANDS) on demand.
- backtest: run a simple long-only backtest with async job pattern and streaming logs.
- portfolio: basic optimizers (equal weight, volatility-inverse placeholder).
- risk: compute rolling volatility, max drawdown, mean return, Calmar ratio, risk-adjusted score (used across scoring paths).
- ml: baseline momentum predictor stub; pluggable for custom models.
- recommendation: aggregate per-symbol scores into ratings and simple trade plans (TP/SL/horizon) given risk appetite.

Backwards compatibility
- All v1 endpoints remain unchanged. v2 endpoints live under /api/v2 and can be enabled per environment via feature flags.

Feature flags (default: all disabled)
- ENABLE_FACTORS=false
- ENABLE_BACKTEST=false
- ENABLE_PORTFOLIO=false
- ENABLE_RISK=false
- ENABLE_ML=false
- ENABLE_RECO=false

How to enable
- .env file: set the flags to true (see .env.example)
- Shell: export ENABLE_FACTORS=true ENABLE_BACKTEST=true ...
- Docker Compose: add the flags under the app service environment.

Dependencies
- Redis (recommended): caching for factor computations and general use. If not available, the cache falls back to in-memory.
  - Configure via environment: REDIS_URL or REDIS_HOST/REDIS_PORT/REDIS_DB. Health probe at GET /api/v2/health/cache.
- Worker: backtests are executed asynchronously using an in-process ThreadPool (no external worker required). For heavier loads you can deploy a separate worker system (e.g., Celery or RQ) in the future; current release runs jobs within the API process with streaming progress.
- Optional MLflow: if you operate custom ML models, you may integrate MLflow for model registry/tracking outside of this service. Not required for the baseline v2 features.

Docker Compose (local with Redis)
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

Resource hints
- Small dev box: 2 vCPU, 2 GB RAM for the app; Redis 256–512 MB. Backtests are CPU-bound; scale threads or instances for throughput.
- Disk: ~200 MB for logs and optional snapshots; more if you enable DuckDB snapshots under data/.
- Network: Akshare data fetches are I/O-bound; consider local mirrors in restricted environments.

API usage (all under /api/v2)
- List available factors
```
curl -s http://localhost:8888/api/v2/factors
```

- Compute factors
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

- Run a backtest (async job pattern)
```
# 1) Submit job
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

# 2) Poll job status/result
curl -s http://localhost:8888/api/v2/backtest/<job_id>

# 3) Stream logs/progress (server-sent json lines)
curl -s -N http://localhost:8888/api/v2/backtest/<job_id>/stream
```

- Portfolio optimize
```
curl -s -X POST http://localhost:8888/api/v2/portfolio/optimize \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL","MSFT","GOOG"], "method": "equal"}'
```

- ML predict (baseline momentum)
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

- Signals recommend
```
curl -s -X POST http://localhost:8888/api/v2/signals/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL","MSFT"],
    "risk_appetite": "balanced",
    "scores": {"AAPL": 0.8, "MSFT": 0.2}
  }'
```

Migrations
- Default deployment uses DuckDB/CSV for optional factor snapshots and requires no database migrations.
- If you introduce a SQL database, follow an additive-only migration policy (no breaking changes). Suggested flow:
  - Create migrations: alembic revision -m "add <table/column>"
  - Apply: alembic upgrade head
  - Roll forward only; never drop or rename in-place during release.
  - Stage first: apply migrations in staging, run smoke backtests and factor computes, then promote to prod.

Security and limits
- JWT: When LOGIN_PASSWORD is set, v1 endpoints require obtaining a bearer token from POST /api/login and sending Authorization: Bearer <token>. v2 endpoints are currently system-to-system and do not require JWT by default; protect them via network policy, ingress rules, or a reverse proxy if needed.
- Rate limits: No built-in limiter; apply at Nginx/ingress (e.g., limit_req) or an API gateway.
- Payload caps: Keep symbol lists small for latency (suggest <= 100). Backtests are CPU-bound; prefer <= 20 symbols per job in small instances.
- ML guardrails: The baseline ML endpoint does not call external services. If you swap in custom models, avoid training/inferring on private data that could leak via logs or responses; scrub logs and validate prompts.

Observability
- Prometheus metrics at /metrics (enabled when prometheus_client is installed):
  - api_latency_seconds Histogram labels: path, method, status
  - cache_hits_total and cache_misses_total Counters
  - backtest_job_duration_seconds Histogram
  - process_memory_rss_bytes Gauge
- Suggested Prometheus scrape_config
```
scrape_configs:
  - job_name: stock-scanner
    static_configs:
      - targets: ['localhost:8888']
    metrics_path: /metrics
```

Quickstart (local dev)
- Install
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
- Start Redis (optional, recommended)
```
docker run -d --name redis -p 6379:6379 redis:7-alpine
```
- Enable v2 and run API
```
export ENABLE_FACTORS=true ENABLE_BACKTEST=true ENABLE_PORTFOLIO=true ENABLE_RISK=true ENABLE_ML=true ENABLE_RECO=true
export ENABLE_CACHE=true REDIS_HOST=localhost REDIS_PORT=6379
uvicorn web_server:app --host 0.0.0.0 --port 8888 --reload
```
- Try one end-to-end example
```
curl -s -X POST http://localhost:8888/api/v2/factors/compute \
  -H "Content-Type: application/json" \
  -d '{"symbols":["600519"],"market":"A","factors":[{"id":"sma","params":{"period":20}}]}'
```

OpenAPI
- Swagger UI: http://localhost:8888/docs (look for tag "v2"). ReDoc: http://localhost:8888/redoc
- v1 remains supported and unchanged; v2 endpoints are additive under /api/v2.

Observability note
- If prometheus_client is not installed, /metrics will return an empty payload and all metrics are no-ops.

Caching note
- utils.cache.Cache automatically uses Redis when available, with an in-memory fallback. Includes decorators, a health probe, and Prometheus hit/miss metrics.
