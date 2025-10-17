# Stock Scanner 项目详细分析

## 1. 项目概览

- **项目定位**：一个聚合股票、基金行情与 AI 分析的全栈应用，支持 A 股、港股、美股、ETF、LOF 多市场多标的的技术面评估，并通过可配置的第三方大模型接口给出文本分析。
- **技术栈**：
  - 后端：FastAPI + Uvicorn，异步 I/O 配合 `httpx`、`asyncio`，数据源主要依赖 AkShare。
  - 前端：Vue 3 + TypeScript + Vite + Naive UI，支持桌面与移动端响应式布局。
  - 日志监控：Loguru，自建日志轮转机制。
  - 部署：提供 Dockerfile 与多种 docker-compose 模板，可结合 nginx 反向代理对外提供服务。
- **核心特性**：
  - 支持手动录入或模糊搜索股票/基金代码，批量触发分析。
  - 后端串联数据抓取 → 技术指标计算 → 评分 → AI 语言模型分析，结果通过流式接口返回。
  - 前端实时解析流式数据，渲染卡片或表格视图，支持结果导出、复制。
  - 可选登录（JWT），可配置公告、API KEY、模型等信息。

## 2. 整体架构

```
┌────────────────────────────────────────────────────┐
│                      前端 (Vue)                    │
│  - 登录页(LoginPage)  - 主界面(StockAnalysisApp)   │
│  - API 配置(ApiConfigPanel)  - 搜索/结果组件        │
│  - ApiService 封装 axios/fetch 调用                │
└───────────────▲───────────────────────┬────────────┘
                │REST/流式 JSON          │JWT Bearer
                │                        │
┌───────────────┴───────────────────────▼────────────┐
│                 FastAPI 应用 (web_server.py)         │
│  - /api/analyze 流式分析  - /api/search_* 搜索接口   │
│  - /api/login & JWT  - /api/test_api_connection     │
│  - 静态资源挂载(frontend/dist)                      │
├────────────────────────────────────────────────────┤
│                    服务层 (services/)               │
│  StockAnalyzerService → {                           │
│    StockDataProvider (AkShare)                      │
│    TechnicalIndicator (技术指标)                    │
│    StockScorer (评分)                               │
│    AIAnalyzer (httpx 调用可配置 LLM)                │
│  }                                                  │
│  USStockServiceAsync / FundServiceAsync (检索)      │
└────────────────────────────────────────────────────┘
```

## 3. 后端结构分析

### 3.1 `web_server.py`

- 初始化 FastAPI 应用，启用全量 CORS，加载 `.env`，通过 `utils.logger` 输出日志。
- JWT 认证：动态生成或使用 `JWT_SECRET_KEY`，`LOGIN_PASSWORD` 为空时可匿名访问，否则 `/api/login` 返回 token，`OptionalOAuth2PasswordBearer` 在“无需登录”模式下短路。
- 公共接口：
  - `POST /api/login`：密码校验，签发一周有效的 access token。
  - `GET /api/check_auth`：校验 token，有效时返回用户名。
  - `GET /api/need_login` / `GET /api/config`：向前端返回是否需要登录、默认公告/模型/超时时间。
  - `POST /api/test_api_connection`：调用 `APIUtils.format_api_url` 规范化地址，通过 `httpx.AsyncClient` 利用调用方配置的 key/model 发送一条测试消息，返回成功/失败。
- 核心流式分析 `POST /api/analyze`：
  - 读取 `AnalyzeRequest`，后端再去重代码。
  - 每次请求构造新的 `StockAnalyzerService`，将前端传入的自定义 API 配置透传给 AI Analyzer。
  - 根据股票数量选择“单股”或“批量”流式模式，使用 `StreamingResponse` 包装异步生成器。
- 辅助接口：
  - `GET /api/search_us_stocks`、`GET /api/us_stock_detail/{symbol}`：调用 `USStockServiceAsync`。
  - `GET /api/search_funds`、`GET /api/fund_detail/{symbol}`：调用 `FundServiceAsync`。
- 静态文件：若存在 `frontend/dist`，将其挂载到根路径，实现单体部署。

### 3.2 分析服务门面 `StockAnalyzerService`

- 构造时组合四个子组件：
  - `StockDataProvider`：封装 AkShare 同步抓取，利用 `asyncio.to_thread` 异步化。
  - `TechnicalIndicator`：计算 MA、RSI、MACD、布林带、ATR、成交量均线/比率、波动率等。
  - `StockScorer`：依据指标（均线结构、RSI 区间、MACD、量能）打分并给出推荐标签。
  - `AIAnalyzer`：封装流式/非流式大模型调用，支持多市场 prompt 模板。
- `analyze_stock`：
  - 拉取单只股票数据，处理异常/空数据。
  - 计算指标、评分、推荐，封装基本结果发送给前端。
  - 继续调用 `AIAnalyzer.get_ai_analysis` 拉取语言模型输出（流式 chunk 或一次性结果）。
- `scan_stocks`（批量）：
  - 并发获取多标的数据 → 计算指标 → 批量评分。
  - 对每个标的即时推送基础信息；若开启流式，再对评分前五的标的做 AI 深入分析。
  - 结束时推送 `scan_completed` 元数据。

### 3.3 数据抓取 `StockDataProvider`

- 针对不同市场调用 AkShare：
  - A 股：`stock_zh_a_hist`（前复权），并统一列名。
  - 港股：`stock_hk_daily`，根据日期过滤；
  - 美股：`stock_us_daily`，补算 `amount`，列名转小写后映射到统一结构；
  - ETF/LOF：分别调用 `fund_etf_hist_em`、`fund_lof_hist_em`。
- 统一输出以日期为索引的 DataFrame，含 Open/High/Low/Close/Volume/Amount 等列，异常时返回带自定义 `error` 属性的空 DataFrame，避免上层崩溃。
- 提供 `get_multiple_stocks_data` 使用信号量限制并发，批量抓取数据集。

### 3.4 技术指标与评分

- `TechnicalIndicator`
  - 采用滚动窗口计算 MA5/20/60、布林带、Volume_MA、Volume_Ratio、RSI、MACD、ATR、20 日波动率等。
  - 内部辅助函数（EMA、RSI、MACD、布林带、ATR）均基于 pandas 实现。
- `StockScorer`
  - 根据多头排列、RSI 区间、MACD、成交量因子累计得分，区间映射推荐语（强烈推荐 → 强烈不推荐）。
  - 提供 `batch_score_stocks`，返回按照得分排序的 (code, score, recommendation) 列表。

### 3.5 AI 分析 `AIAnalyzer`

- 读取 `.env` 或外部传入的 API 配置，利用 `APIUtils.format_api_url` 处理尾缀规则（`/` → `chat/completions`，`#` 表示使用原 URL，默认补 `/v1/chat/completions`）。
- 构造多市场专属 prompt，包含技术摘要（trend、volatility、volume_trend、RSI 等）及最近 14 日数据。
- 调用流程：
  - 先向前端推送一条“正在分析”状态（包含关键指标快照）。
  - **流式模式**：`httpx.AsyncClient().stream` 逐行读取 SSE 风格响应，过滤 `data:`、`[DONE]`，解析 JSON chunk，实时推送 `ai_analysis_chunk` 给前端；结束后尝试提取“投资建议”并计算综合评分。
  - **非流式模式**：一次性 POST 获取完整结果。
  - 发生异常时返回包含 `error` 的 JSON，避免崩溃。
- `_extract_recommendation` 使用正则从 Markdown 中提取“投资建议”段落；`_calculate_analysis_score` 根据技术摘要与文本关键词修正分值。

### 3.6 市场搜索服务

- `USStockServiceAsync`：
  - 缓存美股全量列表（`ak.stock_us_spot_em`），模糊匹配名称、限制返回前 10 项。
  - `get_us_stock_detail` 精确匹配 symbol，返回价格、涨跌幅、估值等。
- `FundServiceAsync`：
  - 分别缓存 ETF 与 LOF 列表（默认 30 分钟），支持代码/名称模糊匹配，返回价格、成交量、市值、折价率等字段。
  - `get_fund_detail` 类似处理。

### 3.7 工具与日志

- `utils/logger.py`：封装 Loguru，创建专用日志目录，分别输出控制台、日常日志、错误日志并保留 7 天；启动时自动清理过期日志。
- `utils/api_utils.py`：单一的 `APIUtils.format_api_url`，供前后端共享 URL 生成规则。

### 3.8 配置与测试

- `.env.example`：列出可配置项（API_KEY、API_URL、API_MODEL、API_TIMEOUT、ANNOUNCEMENT_TEXT、LOGIN_PASSWORD）。
- `requirements.txt`：锁定 Python 依赖版本（AkShare、FastAPI、httpx、python-jose 等）。
- `tests/`：
  - `test_stream.py`：以 `requests` 测试流式大模型接口连通性。
  - `test_akshare.py`：打印 AkShare 版本并测试 `stock_us_daily`。

## 4. 前端结构分析

### 4.1 入口与路由

- `main.ts`：安装 Vue Router、Naive UI，挂载 `App.vue`。
- `App.vue`：包裹多层 Provider（消息、Loading Bar、Dialog、Notification），方便全局调用。
- `router/index.ts`：
  - `createWebHashHistory`，定义 `/` → `StockAnalysisApp`、`/login` → `LoginPage`。
  - 全局前置守卫：
    - 若目标路由 `requiresAuth`，先请求 `/api/need_login` 判断是否启用登录；若需要，检查本地 token 并调用 `/api/check_auth` 校验，否则跳转登录页。

### 4.2 API 封装 `services/api.ts`

- 使用 axios 创建 `axiosInstance`，统一前缀 `/api`。
- 请求拦截器自动携带 token，响应拦截器遇到 401 清除 token。
- 主要方法：
  - `login` / `checkAuth` / `logout`。
  - `analyzeStocks`（仅返回 axios Promise，实际流式读取在组件中通过 `fetch` 完成）。
  - `testApiConnection`、`searchUsStocks`、`searchFunds`、`getConfig`、`checkNeedLogin`。

### 4.3 类型定义与工具

- `types/index.ts`：集中定义 API 请求响应、Stream 消息、前端状态（`StockInfo`、`ApiConfig` 等）。
- `utils/index.ts`：
  - `debounce`、`formatMarketValue`、`parseMarkdown`（依赖 `marked`）、市场开闭市时间计算。
  - 本地存储 API 配置的读写操作。
- `utils/stockValidator.ts`：根据市场类型对输入代码做格式校验（目前 A 股校验占位返回 true，可作为改进点）。

### 4.4 主要组件

- **`StockAnalysisApp.vue`**：
  - 负责主界面布局（左侧参数、右侧结果视图），支持卡片/表格切换、结果导出（CSV/Excel/PDF 占位待实现）。
  - 维护 `marketType`、`stockCodes`、`analyzedStocks`、`apiConfig`、`displayMode` 等状态。
  - 通过 `fetch('/api/analyze')` 获取 ReadableStream，利用 `TextDecoder` + 行缓冲解析后调用 `processStreamData`，分别处理初始化消息、更新消息、完结消息。
  - `handleStreamUpdate` 内部按照字段（`price`, `price_change_value`, `change_percent`, `score`, `ai_analysis_chunk` 等）增量更新对应股票。
  - `copyAnalysisResults` 支持格式化文本复制；导出功能根据 `exportOptions` 提供钩子。
  - 本地存储的 API 配置可通过 `ApiConfigPanel` 回填，公告由 `/api/config` 下发。
- **`ApiConfigPanel.vue`**：
  - 可折叠卡片，展示 API URL/Key/模型/超时时间字段，集成常见模型下拉与快速标签。
  - 支持本地保存配置、`测试连接` 调用 `apiService.testApiConnection`。
  - 将表单改动通过 `update:api-config` 事件抛给父组件。
- **`StockCard.vue`**：
  - 展示单个标的的实时状态、得分、推荐、技术指标摘要，集成 Markdown 渲染分析文本，支持复制、折叠、切换主题色等。
- **`StockSearch.vue`**：
  - 根据市场类型调用 `searchUsStocks` 或 `searchFunds`，带有防抖、加载状态和结果列表，点击后回传 symbol。
- **`MarketTimeDisplay.vue`**：
  - 展示当前时间与中/港/美股市场的开闭状态，依据 `utils.updateMarketTimeInfo` 每分钟刷新。
- **`AnnouncementBanner.vue`**：
  - 展示公告文本，支持倒计时自动关闭、手动关闭、移动端自适应。
- **`LoginPage.vue`**：
  - 单字段密码表单，提交后调用 `apiService.login`，成功写入 token 并跳转首页。

### 4.5 样式与响应式

- 全局样式定义在 `src/style.css`，通过 `.mobile-*` class 名配合 Naive UI 实现移动端适配。
- 组件内部大量使用 `n-grid`、`n-space` 控制布局，并通过 `window.innerWidth` 判断是否切换移动端行为。

## 5. 数据流与交互流程

1. 用户在 `StockAnalysisApp` 选择市场、输入或搜索股票代码，前端使用 `stockValidator` 对输入做基础格式校验。
2. 点击“开始分析”后构造请求体：
   - `stock_codes` 数组（去重）、`market_type` 字段。
   - 若在 `ApiConfigPanel` 设置了自定义 API，则附加 `api_url/api_key/api_model/api_timeout`。
   - 请求头携带 JWT（若开启登录）。
3. 后端 `POST /api/analyze`：
   - 新建 `StockAnalyzerService`，串联数据抓取、指标计算、评分。
   - 先推送基础分析结果，随后由 `AIAnalyzer` 持续推送大模型生成的 Markdown 片段。
4. 浏览器以流式方式读取响应：
   - 第一条消息用于初始化列表（单股或多股）。
   - 后续消息更新各项指标、状态、分析文本，直至收到 `status=completed` / `scan_completed`。
5. 分析结束后，前端可复制文本结果或导出数据；若需要再次分析，可重新触发流程。

## 6. 部署与运行

- Python 依赖通过 `requirements.txt` 安装，FastAPI 可直接运行 `uvicorn web_server:app`；项目提供 `Dockerfile` 以及三份 docker-compose（本地、生产、simple）模板。
- `.env` 用于配置默认公告、API KEY、登录密码等，前后端均会读取。
- 构建前端：在 `frontend` 目录执行 `npm install && npm run build`，生成 `frontend/dist` 后由 FastAPI 静态挂载。
- 若按 docker-compose 方式部署，可同时拉起应用与 nginx，nginx 目录包含默认配置及 SSL 证书占位。

## 7. 改进建议与潜在关注点

1. **A 股代码校验逻辑缺失**：`validateAStock` 目前恒为 `true`，可补充 6 位数字 + 交易所前缀等规则。
2. **前后端配置统一性**：目前 API URL 格式化逻辑在前后端分别实现，功能一致但可考虑抽离共享（例如通过 OpenAPI 返回格式化结果）。
3. **导出功能待完善**：`StockAnalysisApp` 中导出 CSV/Excel/PDF 仅留有菜单入口，可实现实际导出逻辑。
4. **错误处理**：批量分析时，如 AkShare 某只股票报错仅返回错误消息，没有将该标的从列表中剔除，前端仍显示“等待”。可扩展错误标记呈现方式。
5. **依赖稳定性**：AkShare 接口对网络环境、频率敏感，生产环境建议添加缓存或重试策略。
6. **安全性**：`LOGIN_PASSWORD` 为空即放开所有接口，若面向公网部署需确保关闭匿名模式；同时前端本地存储 API Key，需提醒用户风险。
7. **测试覆盖**：当前仅有基础连通性测试，若需持续迭代可补充服务层单元测试与前端组件测试。

## 8. 快速参考

- 后端启动（开发）：
  ```bash
  uvicorn web_server:app --reload --host 0.0.0.0 --port 8888
  ```
- 前端开发：
  ```bash
  cd frontend
  npm install
  npm run dev
  ```
- 构建与打包：
  ```bash
  npm run build   # 生成 dist，供 FastAPI 静态服务
  ```

以上为 stock-scanner 仓库的结构与功能概览，可作为后续维护、二次开发或排障的参考文档。
