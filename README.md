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

New non-breaking modules are added under /api/v2 and are disabled by default via feature flags. Enable them in environment by setting the following to true:
- ENABLE_FACTORS
- ENABLE_BACKTEST
- ENABLE_PORTFOLIO
- ENABLE_RISK
- ENABLE_ML
- ENABLE_RECO

Endpoints (all under /api/v2):
- GET /factors -> list available TA factors
- POST /factors/compute -> compute SMA/EMA/RSI/MACD/ATR/BBANDS for given symbols and range
- POST /backtest/run -> enqueue a simple long-only backtest; returns job_id
- GET /backtest/{job_id} -> job status/result
- GET /backtest/{job_id}/stream -> streaming progress logs
- POST /portfolio/optimize -> lightweight equal/vol-inverse weights
- POST /ml/predict -> baseline momentum predictor stub
- POST /signals/recommend -> aggregate scores to rating and trade plan

Observability:
- Prometheus metrics endpoint exposed at /metrics (API latency histograms, cache metrics, backtest durations; auto no-op if prometheus_client not installed).

Caching:
- New cache wrapper utils.cache.Cache with Redis pool support, in-memory fallback, decorators, health probe, and Prometheus hit/miss metrics.
