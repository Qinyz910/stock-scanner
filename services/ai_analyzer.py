import pandas as pd
import os
import json
import httpx
import re
import asyncio
import time
import random
from typing import AsyncGenerator
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.api_utils import APIUtils
from utils import metrics
from utils.rate_limiter import RateLimiter
from utils.cache import Cache
from datetime import datetime
from urllib.parse import urlparse

# 获取日志器
logger = get_logger()


def generate_placeholder(stock_code: str, technical_summary: dict) -> str:
    try:
        trend = technical_summary.get('trend')
        vol = technical_summary.get('volatility')
        vol_tr = technical_summary.get('volume_trend')
        rsi = technical_summary.get('rsi_level')
        parts = [
            f"标的 {stock_code} 占位分析：",
            f"趋势:{'上行' if trend=='upward' else '下行'} | 波动率:{vol} | 成交量趋势:{'放大' if vol_tr=='increasing' else '缩小'} | RSI:{rsi}",
            "建议：数据受限/配额紧张，暂不进行激进判断，保持观望或轻仓，待稍后重试获取完整分析。",
        ]
        return "\n".join(parts)
    except Exception:
        return f"{stock_code} 占位分析：当前AI服务暂不可用，请稍后重试。"


class AIAnalyzer:
    """
    异步AI分析服务
    负责调用AI API对股票数据进行分析
    """

    def __init__(self, custom_api_url=None, custom_api_key=None, custom_api_model=None, custom_api_timeout=None):
        """
        初始化AI分析服务

        Args:
            custom_api_url: 自定义API URL
            custom_api_key: 自定义API密钥
            custom_api_model: 自定义API模型
            custom_api_timeout: 自定义API超时时间
        """
        # 加载环境变量
        load_dotenv()

        # 统一解析环境配置（允许自定义参数覆盖）
        conf = APIUtils.resolve_ai_config()
        # Provider 类型（行为适配）：newapi|openai|deepseek|gemini，默认 newapi（OpenAI 兼容）
        try:
            self.AI_PROVIDER = (os.getenv('AI_PROVIDER', '') or '').lower().strip() or 'newapi'
        except Exception:
            self.AI_PROVIDER = 'newapi'

        # 设置API配置（自定义优先，其次解析结果）
        self.API_URL = custom_api_url or conf.get('base_url')
        if self.AI_PROVIDER == 'gemini':
            self.API_KEY = custom_api_key or os.getenv('GEMINI_API_KEY') or os.getenv('AI_API_KEY') or os.getenv('API_KEY')
        else:
            self.API_KEY = custom_api_key or os.getenv('AI_API_KEY') or os.getenv('API_KEY')
        # 模型不设置默认值，保持环境解析与自定义为准
        self.API_MODEL = custom_api_model or conf.get('model') or os.getenv('API_MODEL')
        self.API_TIMEOUT = int(custom_api_timeout or os.getenv('API_TIMEOUT', 60))
        self.API_MAX_TOKENS = int(os.getenv('API_MAX_TOKENS', '1024'))

        # Provider标识用于限流/指标（基于URL host）
        try:
            _u = urlparse(self.API_URL or "")
            self.PROVIDER = _u.hostname or (_u.netloc or "custom")
        except Exception:
            self.PROVIDER = "custom"

        # 结果缓存（命名空间按模型区分）
        self.cache_ttl = int(os.getenv("AI_CACHE_TTL", "1200"))
        self.cache = Cache(namespace=f"ai_results_{self.API_MODEL}")

        # 输出完整性与分块参数
        self.MIN_OUTPUT_CHARS = int(os.getenv("AI_MIN_OUTPUT_CHARS", "600"))
        self.CHUNK_MAX_BYTES = int(os.getenv("AI_CHUNK_MAX_BYTES", "3072"))
        self.CHUNK_MIN_SENTENCES = int(os.getenv("AI_CHUNK_MIN_SENTENCES", "2"))
        self.REQUIRED_SECTIONS = ["概览", "主要信号", "技术面", "基本面", "风险", "操作建议"]

        # SSE/streaming timeouts
        try:
            self.SSE_IDLE_SECONDS = float(os.getenv("AI_SSE_IDLE_SECONDS", "8"))
        except Exception:
            self.SSE_IDLE_SECONDS = 8.0
        try:
            self.SSE_TOTAL_SECONDS = float(os.getenv("AI_SSE_TOTAL_SECONDS", "30"))
        except Exception:
            self.SSE_TOTAL_SECONDS = 30.0
        try:
            self.SSE_TTFB_SECONDS = float(os.getenv("AI_SSE_TTFB_SECONDS", "3"))
        except Exception:
            self.SSE_TTFB_SECONDS = 3.0
        try:
            self.MAX_PROVIDER_SWITCHES = int(os.getenv("AI_MAX_PROVIDER_SWITCHES", "1"))
        except Exception:
            self.MAX_PROVIDER_SWITCHES = 1

        # 备用提供商配置（主->备1->备2）
        self.BACKUP_PROVIDERS = []
        try:
            b1_url = os.getenv("API_URL_BACKUP1")
            b1_key = os.getenv("API_KEY_BACKUP1")
            b1_model = os.getenv("API_MODEL_BACKUP1") or self.API_MODEL
            if b1_url and b1_key:
                self.BACKUP_PROVIDERS.append({"url": b1_url, "key": b1_key, "model": b1_model})
        except Exception:
            pass
        try:
            b2_url = os.getenv("API_URL_BACKUP2")
            b2_key = os.getenv("API_KEY_BACKUP2")
            b2_model = os.getenv("API_MODEL_BACKUP2") or self.API_MODEL
            if b2_url and b2_key:
                self.BACKUP_PROVIDERS.append({"url": b2_url, "key": b2_key, "model": b2_model})
        except Exception:
            pass

        logger.debug(
            f"初始化AIAnalyzer: PROVIDER={self.PROVIDER}, API_URL={self.API_URL}, API_MODEL={self.API_MODEL}, API_KEY={'已提供' if self.API_KEY else '未提供'}, API_TIMEOUT={self.API_TIMEOUT}, "
            f"MIN_CHARS={self.MIN_OUTPUT_CHARS}, CHUNK_MAX_BYTES={self.CHUNK_MAX_BYTES}, CHUNK_MIN_SENTENCES={self.CHUNK_MIN_SENTENCES}, SSE_IDLE_SECONDS={self.SSE_IDLE_SECONDS}, SSE_TOTAL_SECONDS={self.SSE_TOTAL_SECONDS}, SSE_TTFB_SECONDS={self.SSE_TTFB_SECONDS}, MAX_SWITCHES={self.MAX_PROVIDER_SWITCHES}, BACKUPS={len(self.BACKUP_PROVIDERS)}"
        )

    async def get_ai_analysis(self, df: pd.DataFrame, stock_code: str, market_type: str = 'A', stream: bool = False) -> AsyncGenerator[str, None]:
        """
        对股票数据进行AI分析

        Args:
            df: 包含技术指标的DataFrame
            stock_code: 股票代码
            market_type: 市场类型，默认为'A'股
            stream: 是否使用流式响应

        Returns:
            异步生成器，生成分析结果字符串
        """
        try:
            start_time = time.perf_counter()
            logger.info(f"开始AI分析 {stock_code}, 流式模式: {stream}")

            # 基础配置校验（避免 NameError 等因未初始化变量导致的错误）
            if not self.API_URL or not self.API_KEY or not self.API_MODEL:
                raise ValueError(f"AI 配置不完整: provider={self.AI_PROVIDER}, model={self.API_MODEL}, base_url={'set' if self.API_URL else 'missing'}, api_key_present={bool(self.API_KEY)}")

            # 提取关键技术指标
            if df is None or df.empty:
                msg = "暂无可用分析素材(no_data)"
                logger.warning(f"{stock_code} {msg}")
                yield json.dumps({
                    "stock_code": stock_code,
                    "status": "completed",
                    "analysis": "暂无可用分析素材",
                    "recommendation": "观望",
                    "score": 50
                })
                try:
                    metrics.record_ai_stream_zero_chunks(self.API_MODEL, reason="no_data")
                    metrics.observe_ai_stream_duration(0.0, model=self.API_MODEL, outcome="no_data")
                except Exception:
                    pass
                return
            latest_data = df.iloc[-1]

            # 计算技术指标
            rsi = latest_data.get('RSI')
            price = latest_data.get('Close')
            price_change = latest_data.get('Change')

            # 确定MA趋势
            ma_trend = 'UP' if latest_data.get('MA5', 0) > latest_data.get('MA20', 0) else 'DOWN'

            # 确定MACD信号
            macd = latest_data.get('MACD', 0)
            macd_signal = latest_data.get('MACD_Signal', 0)
            macd_signal_type = 'BUY' if macd > macd_signal else 'SELL'

            # 确定成交量状态
            volume_ratio = latest_data.get('Volume_Ratio', 1)
            volume_status = 'HIGH' if volume_ratio > 1.5 else ('LOW' if volume_ratio < 0.5 else 'NORMAL')

            # AI 分析内容
            recent_data = df.tail(14).to_dict('records')
            technical_summary = {
                'trend': 'upward' if df.iloc[-1]['MA5'] > df.iloc[-1]['MA20'] else 'downward',
                'volatility': f"{df.iloc[-1]['Volatility']:.2f}%",
                'volume_trend': 'increasing' if df.iloc[-1]['Volume_Ratio'] > 1 else 'decreasing',
                'rsi_level': df.iloc[-1]['RSI']
            }

            # 根据市场类型调整分析提示
            if market_type in ['ETF', 'LOF']:
                prompt = f"""
                分析基金 {stock_code}：

                技术指标概要：
                {technical_summary}
                
                近14日交易数据：
                {recent_data}
                
                请提供：
                1. 净值走势分析（包含支撑位和压力位）
                2. 成交量分析及其对净值的影响
                3. 风险评估（包含波动率和折溢价分析）
                4. 短期和中期净值预测
                5. 关键价格位分析
                6. 申购赎回建议（包含止损位）
                
                请基于技术指标和市场表现进行分析，给出具体数据支持。
                """
            elif market_type == 'US':
                prompt = f"""
                分析美股 {stock_code}：

                技术指标概要：
                {technical_summary}
                
                近14日交易数据：
                {recent_data}
                
                请提供：
                1. 趋势分析（包含支撑位和压力位，美元计价）
                2. 成交量分析及其含义
                3. 风险评估（包含波动率和美股市场特有风险）
                4. 短期和中期目标价位（美元）
                5. 关键技术位分析
                6. 具体交易建议（包含止损位）
                
                请基于技术指标和美股市场特点进行分析，给出具体数据支持。
                """
            elif market_type == 'HK':
                prompt = f"""
                分析港股 {stock_code}：

                技术指标概要：
                {technical_summary}
                
                近14日交易数据：
                {recent_data}
                
                请提供：
                1. 趋势分析（包含支撑位和压力位，港币计价）
                2. 成交量分析及其含义
                3. 风险评估（包含波动率和港股市场特有风险）
                4. 短期和中期目标价位（港币）
                5. 关键技术位分析
                6. 具体交易建议（包含止损位）
                
                请基于技术指标和港股市场特点进行分析，给出具体数据支持。
                """
            else:  # A股
                prompt = f"""
                分析A股 {stock_code}：

                技术指标概要：
                {technical_summary}
                
                近14日交易数据：
                {recent_data}
                
                请提供：
                1. 趋势分析（包含支撑位和压力位）
                2. 成交量分析及其含义
                3. 风险评估（包含波动率分析）
                4. 短期和中期目标价位
                5. 关键技术位分析
                6. 具体交易建议（包含止损位）
                
                请基于技术指标和A股市场特点进行分析，给出具体数据支持。
                """

            # 格式化API URL（根据 provider 与是否流式选择合适的端点）
            api_url = APIUtils.format_ai_url(self.API_URL, model=self.API_MODEL, provider=self.AI_PROVIDER, stream=bool(stream))

            # 准备请求数据（禁用工具调用，控制生成，避免空输出）
            request_data = {
                "model": self.API_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": self.API_MAX_TOKENS,
                "tools": [],
                "tool_choice": "none",
                "stream": stream
            }

            # 准备请求头（根据 provider 选择鉴权方式）
            if (self.AI_PROVIDER or "") == "gemini":
                headers_base = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": f"{self.API_KEY}",
                    "User-Agent": f"stock-scanner/1.0 (+https://github.com/) model/{self.API_MODEL} provider/{self.PROVIDER}",
                }
                headers_stream = dict(headers_base)
                headers_stream.update({
                    "Accept": "application/json",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                })
                headers_json = dict(headers_base)
                headers_json.update({
                    "Accept": "application/json"
                })
            else:
                headers_base = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.API_KEY}",
                    "User-Agent": f"stock-scanner/1.0 (+https://github.com/) model/{self.API_MODEL} provider/{self.PROVIDER}",
                }
                headers_stream = dict(headers_base)
                headers_stream.update({
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                })
                headers_json = dict(headers_base)
                headers_json.update({
                    "Accept": "application/json"
                })

            # 获取当前日期作为分析日期
            analysis_date = datetime.now().strftime("%Y-%m-%d")

            # 统一限流器（provider+model级）与相关参数
            limiter = RateLimiter.get(self.PROVIDER, self.API_MODEL)
            correlation_id = f"ai-{stock_code}-{int(time.time())}-{random.randint(1000,9999)}"

            # 结果缓存命中则直接返回（减少上游压力）
            cache_key = json.dumps({
                "sym": stock_code,
                "date": analysis_date,
                "market": market_type,
                "model": self.API_MODEL,
                "provider": self.PROVIDER,
                "tpl": "v1"
            }, ensure_ascii=False, sort_keys=True)
            cached = self.cache.get(cache_key)
            if cached and isinstance(cached, dict) and cached.get("analysis_text"):
                analysis_text = cached.get("analysis_text", "")
                # 先发一条chunk，保持与流式兼容
                yield json.dumps({
                    "stock_code": stock_code,
                    "ai_analysis_chunk": analysis_text,
                    "status": "analyzing"
                })
                # 完成事件
                recommendation = self._extract_recommendation(analysis_text)
                score = self._calculate_analysis_score(analysis_text, technical_summary)
                yield json.dumps({
                    "stock_code": stock_code,
                    "status": "completed",
                    "score": score,
                    "recommendation": recommendation
                })
                return

            # 异步请求API
            timeout_post = httpx.Timeout(read=self.API_TIMEOUT, connect=30.0, write=self.API_TIMEOUT, pool=self.API_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout_post) as client:
                logger.debug(f"[{correlation_id}] 发送AI请求: URL={api_url}, MODEL={self.API_MODEL}, STREAM={stream}")

                # 先发送技术指标数据
                yield json.dumps({
                    "stock_code": stock_code,
                    "status": "analyzing",
                    "rsi": rsi,
                    "price": price,
                    "price_change": price_change,
                    "ma_trend": ma_trend,
                    "macd_signal": macd_signal_type,
                    "volume_status": volume_status,
                    "analysis_date": analysis_date
                })

                if stream:
                    # Provider chain: primary then backups
                    providers = []
                    providers.append({
                        "url": self.API_URL,
                        "key": self.API_KEY,
                        "model": self.API_MODEL,
                        "provider_name": self.PROVIDER,
                    })
                    # Extend with configured backups
                    for b in (self.BACKUP_PROVIDERS or []):
                        try:
                            _u = urlparse(b.get("url") or "")
                            pname = _u.hostname or (_u.netloc or "backup")
                        except Exception:
                            pname = "backup"
                        providers.append({
                            "url": b.get("url"),
                            "key": b.get("key"),
                            "model": b.get("model") or self.API_MODEL,
                            "provider_name": pname,
                        })

                    first_chunk_snapshot = None
                    last_chunk_snapshot = None
                    chunk_count = 0
                    chunks_sent = 0
                    buffer = ""
                    pending = ""
                    saw_finish_reason = None

                    switches = 0
                    outcome = "ok"
                    full_content = ""
                    switch_reason = None

                    # Iterate providers
                    current_model = self.API_MODEL  # 明确初始化，避免未赋值引用
                    for idx, pv in enumerate(providers):
                        if idx > 0:
                            # Announce provider switch before starting
                            yield json.dumps({
                                "stock_code": stock_code,
                                "event": "provider_switch",
                                "from": {
                                    "provider": providers[idx-1]["provider_name"],
                                    "model": providers[idx-1]["model"],
                                },
                                "to": {
                                    "provider": pv["provider_name"],
                                    "model": pv["model"],
                                },
                                "reason": switch_reason,
                            })
                            try:
                                metrics.record_ai_provider_switch(
                                    providers[idx-1]["provider_name"], providers[idx-1]["model"], pv["provider_name"], pv["model"], switch_reason or "unknown"
                                )
                            except Exception:
                                pass

                        current_base_url = pv["url"]
                        current_model = pv["model"]
                        current_key = pv["key"]
                        current_provider_name = pv["provider_name"]
                        # 依据 AI_PROVIDER 生成请求地址
                        current_url = APIUtils.format_ai_url(current_base_url, model=current_model, provider=self.AI_PROVIDER, stream=True)
                        # Recompute limiter per provider/model
                        limiter = RateLimiter.get(current_provider_name, current_model)

                        # Prepare headers per provider
                        if (self.AI_PROVIDER or "") == "gemini":
                            headers_base = {
                                "Content-Type": "application/json",
                                "x-goog-api-key": f"{current_key}",
                                "User-Agent": f"stock-scanner/1.0 (+https://github.com/) model/{current_model} provider/{current_provider_name}",
                            }
                            headers_stream = dict(headers_base)
                            headers_stream.update({
                                # Gemini 官方接口通常返回 application/json 流
                                "Accept": "application/json",
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            })
                            headers_json = dict(headers_base)
                            headers_json.update({"Accept": "application/json"})
                        else:
                            headers_base = {
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {current_key}",
                                "User-Agent": f"stock-scanner/1.0 (+https://github.com/) model/{current_model} provider/{current_provider_name}",
                            }
                            headers_stream = dict(headers_base)
                            headers_stream.update({
                                "Accept": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "Connection": "keep-alive",
                            })
                            headers_json = dict(headers_base)
                            headers_json.update({"Accept": "application/json"})

                        # Request payload per provider
                        request_data = {
                            "model": current_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.7,
                            "max_tokens": self.API_MAX_TOKENS,
                            "tools": [],
                            "tool_choice": "none",
                            "stream": True,
                        }

                        # SSE state per provider attempt
                        max_attempts = 3
                        attempt = 0
                        sse_line_buffer = ""
                        sse_data_lines = []
                        sse_event_name = None
                        ttfb_recorded = False
                        ttfb_seconds = 0.0
                        sample_logged = False
                        fallback_reason = None
                        timeout_stream = httpx.Timeout(read=float(getattr(self, 'SSE_IDLE_SECONDS', 8)), connect=30.0, write=30.0, pool=30.0)

                        while attempt < max_attempts:
                            await limiter.wait_for_slot()
                            try:
                                req_start = time.perf_counter()
                                try:
                                    stream_cm = client.stream("POST", current_url, json=request_data, headers=headers_stream, timeout=timeout_stream)
                                except TypeError:
                                    stream_cm = client.stream("POST", current_url, json=request_data, headers=headers_stream)
                                async with stream_cm as response:
                                    try:
                                        logger.debug(f"[{correlation_id}] 上游响应头: {dict(response.headers)}")
                                    except Exception:
                                        pass

                                    if response.status_code != 200:
                                        try:
                                            error_text = await response.aread()
                                            error_data = json.loads(error_text)
                                            error_message = error_data.get('error', {}).get('message', '未知错误')
                                        except Exception:
                                            error_message = f"HTTP {response.status_code}"
                                        logger.error(f"[{correlation_id}] AI 流式请求失败: {response.status_code} - {error_message}")
                                        if response.status_code in (401, 403):
                                            fallback_reason = "401"
                                            limiter.release()
                                            break
                                        if response.status_code in (429, 500, 502, 503, 504):
                                            retry_after = None
                                            try:
                                                retry_after = response.headers.get("Retry-After")
                                            except Exception:
                                                retry_after = None
                                            if response.status_code == 429:
                                                try:
                                                    metrics.record_ai_rate_limit_hit(current_provider_name, current_model)
                                                except Exception:
                                                    pass
                                                fallback_reason = "429"
                                            else:
                                                fallback_reason = "5xx"
                                            delay = await limiter.on_transient_error(response.status_code, retry_after)
                                            try:
                                                metrics.record_ai_retry(current_provider_name, current_model, reason=str(response.status_code))
                                            except Exception:
                                                pass
                                            limiter.release()
                                            await asyncio.sleep(delay)
                                            attempt += 1
                                            continue
                                        else:
                                            limiter.release()
                                            fallback_reason = "5xx"
                                            break
                                    else:
                                        done = False
                                        async for chunk in response.aiter_text():
                                            # total budget check
                                            try:
                                                if float(getattr(self, 'SSE_TOTAL_SECONDS', 0)) > 0 and (time.perf_counter() - start_time) > float(getattr(self, 'SSE_TOTAL_SECONDS', 0)):
                                                    logger.warning(f"[{correlation_id}] 达到端到端超时预算，停止当前流")
                                                    fallback_reason = fallback_reason or "idle_timeout"
                                                    done = True
                                                    break
                                            except Exception:
                                                pass
                                            if not chunk:
                                                continue
                                            sse_line_buffer += chunk
                                            while "\n" in sse_line_buffer:
                                                raw_line, sse_line_buffer = sse_line_buffer.split("\n", 1)
                                                line = raw_line.rstrip("\r")

                                                if line.strip() == "":
                                                    if sse_data_lines:
                                                        payload = "\n".join(sse_data_lines).strip()
                                                        sse_data_lines = []
                                                        if payload:
                                                            if not ttfb_recorded:
                                                                ttfb_seconds = time.perf_counter() - req_start
                                                                try:
                                                                    metrics.observe_ai_upstream_ttfb(current_provider_name, current_model, ttfb_seconds)
                                                                except Exception:
                                                                    pass
                                                                ttfb_recorded = True
                                                            if not sample_logged:
                                                                try:
                                                                    logger.debug(f"[{correlation_id}] 首帧载荷样本: {payload[:1024]}")
                                                                except Exception:
                                                                    pass
                                                                sample_logged = True
                                                            if payload == "[DONE]":
                                                                done = True
                                                                break
                                                            try:
                                                                if payload.strip().startswith('{'):
                                                                    chunk_data = json.loads(payload)
                                                                else:
                                                                    chunk_data = None
                                                                if not chunk_data:
                                                                    continue
                                                                choices_list = chunk_data.get("choices", []) if isinstance(chunk_data, dict) else []
                                                                if not isinstance(choices_list, list) or len(choices_list) == 0:
                                                                    continue
                                                                first_choice = choices_list[0] or {}
                                                                finish_reason = first_choice.get("finish_reason")
                                                                if finish_reason in ("stop", "length"):
                                                                    saw_finish_reason = finish_reason
                                                                    continue
                                                                delta = first_choice.get("delta", {}) or {}
                                                                content = delta.get("content") if isinstance(delta, dict) else None
                                                                if not content:
                                                                    content = first_choice.get("text") or None
                                                                if content:
                                                                    if first_chunk_snapshot is None:
                                                                        first_chunk_snapshot = content[:100]
                                                                    last_chunk_snapshot = content[-100:]
                                                                    chunk_count += 1
                                                                    buffer += content
                                                                    pending += content
                                                                    out_chunks, pending = self._drain_chunks_from_buffer(pending)
                                                                    for c in out_chunks:
                                                                        chunks_sent += 1
                                                                        # emit both legacy and new event
                                                                        yield json.dumps({
                                                                            "stock_code": stock_code,
                                                                            "ai_analysis_chunk": c,
                                                                            "event": "ai_delta",
                                                                            "delta": c,
                                                                            "status": "analyzing"
                                                                        })
                                                            except json.JSONDecodeError:
                                                                logger.error(f"[{correlation_id}] JSON解析错误，事件载荷: {payload[:180]}")
                                                                continue
                                                    continue

                                                if line.startswith("data:"):
                                                    data_val = line[5:]
                                                    if data_val.startswith(" "):
                                                        data_val = data_val[1:]
                                                    if sse_data_lines:
                                                        prev_payload = "\n".join(sse_data_lines).strip()
                                                        if prev_payload:
                                                            if not ttfb_recorded:
                                                                ttfb_seconds = time.perf_counter() - req_start
                                                                try:
                                                                    metrics.observe_ai_upstream_ttfb(current_provider_name, current_model, ttfb_seconds)
                                                                except Exception:
                                                                    pass
                                                                ttfb_recorded = True
                                                            if prev_payload == "[DONE]":
                                                                done = True
                                                                sse_data_lines = []
                                                                break
                                                            try:
                                                                if prev_payload.strip().startswith('{'):
                                                                    chunk_data = json.loads(prev_payload)
                                                                else:
                                                                    chunk_data = None
                                                                if chunk_data:
                                                                    choices_list = chunk_data.get("choices", []) if isinstance(chunk_data, dict) else []
                                                                    if isinstance(choices_list, list) and len(choices_list) > 0:
                                                                        first_choice = choices_list[0] or {}
                                                                        finish_reason = first_choice.get("finish_reason")
                                                                        if finish_reason in ("stop", "length"):
                                                                            saw_finish_reason = finish_reason
                                                                        else:
                                                                            delta = first_choice.get("delta", {}) or {}
                                                                            content = delta.get("content") if isinstance(delta, dict) else None
                                                                            if not content:
                                                                                content = first_choice.get("text") or None
                                                                            if content:
                                                                                if first_chunk_snapshot is None:
                                                                                    first_chunk_snapshot = content[:100]
                                                                                last_chunk_snapshot = content[-100:]
                                                                                chunk_count += 1
                                                                                buffer += content
                                                                                pending += content
                                                                                out_chunks, pending = self._drain_chunks_from_buffer(pending)
                                                                                for c in out_chunks:
                                                                                    chunks_sent += 1
                                                                                    yield json.dumps({
                                                                                        "stock_code": stock_code,
                                                                                        "ai_analysis_chunk": c,
                                                                                        "event": "ai_delta",
                                                                                        "delta": c,
                                                                                        "status": "analyzing"
                                                                                    })
                                                            except json.JSONDecodeError:
                                                                logger.error(f"[{correlation_id}] JSON解析错误，事件载荷: {prev_payload[:180]}")
                                                            finally:
                                                                sse_data_lines = []
                                                    sse_data_lines.append(data_val)
                                                    continue
                                                if line.startswith("event:"):
                                                    sse_event_name = line[6:].strip() or None
                                                    continue
                                            if done:
                                                break

                                        if not done and sse_data_lines:
                                            payload = "\n".join(sse_data_lines).strip()
                                            sse_data_lines = []
                                            if payload:
                                                if not ttfb_recorded:
                                                    ttfb_seconds = time.perf_counter() - req_start
                                                    try:
                                                        metrics.observe_ai_upstream_ttfb(current_provider_name, current_model, ttfb_seconds)
                                                    except Exception:
                                                        pass
                                                    ttfb_recorded = True
                                                if payload != "[DONE]":
                                                    try:
                                                        if payload.strip().startswith('{'):
                                                            chunk_data = json.loads(payload)
                                                        else:
                                                            chunk_data = None
                                                        if chunk_data:
                                                            choices_list = chunk_data.get("choices", []) if isinstance(chunk_data, dict) else []
                                                            if isinstance(choices_list, list) and len(choices_list) > 0:
                                                                first_choice = choices_list[0] or {}
                                                                delta = first_choice.get("delta", {}) or {}
                                                                content = delta.get("content") if isinstance(delta, dict) else None
                                                                if not content:
                                                                    content = first_choice.get("text") or None
                                                                if content:
                                                                    if first_chunk_snapshot is None:
                                                                        first_chunk_snapshot = content[:100]
                                                                    last_chunk_snapshot = content[-100:]
                                                                    chunk_count += 1
                                                                    buffer += content
                                                                    pending += content
                                                                    out_chunks, pending = self._drain_chunks_from_buffer(pending)
                                                                    for c in out_chunks:
                                                                        chunks_sent += 1
                                                                        yield json.dumps({
                                                                            "stock_code": stock_code,
                                                                            "ai_analysis_chunk": c,
                                                                            "event": "ai_delta",
                                                                            "delta": c,
                                                                            "status": "analyzing"
                                                                        })
                                                    except json.JSONDecodeError:
                                                        logger.error(f"[{correlation_id}] JSON解析错误，剩余事件载荷: {payload[:180]}")
                                                        pass

                                await limiter.on_success()
                                limiter.release()
                                # If we got any chunks from this provider, consider success and do not switch further
                                if chunk_count > 0:
                                    try:
                                        metrics.record_ai_provider_switch_success(current_provider_name, current_model)
                                    except Exception:
                                        pass
                                    break
                            except httpx.ReadTimeout:
                                try:
                                    metrics.record_ai_upstream_idle_timeout(current_provider_name, current_model)
                                except Exception:
                                    pass
                                logger.warning(f"[{correlation_id}] 上游流在空闲期间超时 (idle read)")
                                fallback_reason = "idle_timeout"
                                limiter.release()
                                break
                            except httpx.RequestError as e:
                                logger.error(f"[{correlation_id}] 流式请求网络错误: {str(e)}")
                                try:
                                    metrics.record_ai_retry(current_provider_name, current_model, reason="network")
                                except Exception:
                                    pass
                                fallback_reason = fallback_reason or "5xx"
                                limiter.release()
                                await asyncio.sleep(RateLimiter.compute_backoff(attempt))
                                attempt += 1
                                continue

                        # Provider done or failed
                        if chunk_count == 0:
                            # Decide whether to switch
                            if switches < int(getattr(self, 'MAX_PROVIDER_SWITCHES', 1)) and (idx + 1) < len(providers):
                                switches += 1
                                switch_reason = fallback_reason or "no_events"
                                try:
                                    metrics.record_ai_stream_zero_chunks(current_model, reason="zero_chunks")
                                    metrics.record_ai_stream_empty(current_provider_name, current_model)
                                except Exception:
                                    pass
                                # continue to next provider in chain
                                continue
                            else:
                                # no more providers; break and finish
                                outcome = "failed"
                                break
                        else:
                            # success from this provider; continue to finalization
                            outcome = "ok"
                            break

                    # Flush any remainder
                    if pending.strip():
                        yield json.dumps({
                            "stock_code": stock_code,
                            "ai_analysis_chunk": pending,
                            "event": "ai_delta",
                            "delta": pending,
                            "status": "analyzing"
                        })
                        chunks_sent += 1
                        pending = ""

                    # Record fragments metric for final provider (if any)
                    try:
                        final_provider = providers[min(idx, len(providers)-1)]["provider_name"] if providers else self.PROVIDER
                        final_model = providers[min(idx, len(providers)-1)]["model"] if providers else self.API_MODEL
                        metrics.record_ai_stream_fragments(final_provider, final_model, chunk_count)
                        if chunk_count == 0:
                            metrics.record_ai_stream_empty(final_provider, final_model)
                    except Exception:
                        pass

                    full_content = buffer

                    # 若仍未收到任何内容片段，触发一次非流式补救
                    if not full_content:
                        try:
                            metrics.record_ai_stream_zero_then_fallback(final_provider, final_model)
                            metrics.record_ai_fallback_non_stream(final_provider, final_model, reason=fallback_reason or "empty_stream")
                        except Exception:
                            pass
                        # 构造非流式地址与载荷
                        nonstream_url = APIUtils.format_ai_url(current_base_url, model=final_model, provider=self.AI_PROVIDER, stream=False)
                        fb_attempts = 2
                        fb_try = 0
                        analysis_text = ""
                        while fb_try < fb_attempts and not analysis_text:
                            payload = dict(request_data)
                            payload["stream"] = False
                            try:
                                await limiter.wait_for_slot()
                                resp = await client.post(nonstream_url, json=payload, headers=headers_json)
                                if resp.status_code == 200:
                                    data = resp.json()
                                    # OpenAI/newapi 解析
                                    choices_list = data.get("choices", []) if isinstance(data, dict) else []
                                    if isinstance(choices_list, list) and len(choices_list) > 0:
                                        first_choice = choices_list[0] or {}
                                        message = first_choice.get("message", {})
                                        if isinstance(message, dict) and message:
                                            analysis_text = message.get("content", "") or ""
                                        else:
                                            analysis_text = first_choice.get("text", "") or ""
                                    # Gemini 官方解析
                                    if not analysis_text and isinstance(data, dict):
                                        cands = data.get("candidates") or []
                                        if isinstance(cands, list) and cands:
                                            cand0 = cands[0] or {}
                                            content_obj = cand0.get("content") or {}
                                            parts = content_obj.get("parts") or []
                                            if isinstance(parts, list) and parts:
                                                analysis_text = parts[0].get("text") or ""
                                    await limiter.on_success()
                                    limiter.release()
                                elif resp.status_code in (429, 500, 502, 503, 504):
                                    delay = await limiter.on_transient_error(resp.status_code, resp.headers.get("Retry-After") if hasattr(resp, "headers") else None)
                                    limiter.release()
                                    await asyncio.sleep(delay)
                                    fb_try += 1
                                    continue
                                else:
                                    limiter.release()
                                    break
                            except httpx.RequestError:
                                limiter.release()
                                await asyncio.sleep(RateLimiter.compute_backoff(fb_try))
                                fb_try += 1
                                continue
                        if analysis_text:
                            try:
                                metrics.record_ai_fallback_success(final_provider, final_model, reason=fallback_reason or "empty_stream")
                                metrics.record_ai_fallback_bytes(final_provider, final_model, len(analysis_text.encode("utf-8")))
                            except Exception:
                                pass
                            full_content = analysis_text
                            outcome = "degraded"
                            for c in self._chunk_text(analysis_text):
                                chunks_sent += 1
                                yield json.dumps({
                                    "stock_code": stock_code,
                                    "ai_analysis_chunk": c,
                                    "event": "ai_delta",
                                    "delta": c,
                                    "status": "analyzing"
                                })
                        else:
                            try:
                                metrics.record_ai_fallback_fail(final_provider, final_model, reason=fallback_reason or "empty_stream")
                            except Exception:
                                pass

                    # 完整性控制与续写（仅在已有内容时进行）
                    if full_content:
                        missing_sections = self._detect_missing_sections(full_content)
                        if missing_sections:
                            for sec in missing_sections:
                                try:
                                    metrics.record_ai_missing_section(final_provider, final_model, sec)
                                except Exception:
                                    pass
                        if saw_finish_reason == "length" or len(full_content) < self.MIN_OUTPUT_CHARS or missing_sections:
                            try:
                                metrics.record_ai_truncated_response(final_provider, final_model, reason=(saw_finish_reason or ("short" if len(full_content) < self.MIN_OUTPUT_CHARS else "incomplete")))
                            except Exception:
                                pass
                            rounds = 0
                            while rounds < 2 and (len(full_content) < self.MIN_OUTPUT_CHARS or missing_sections):
                                try:
                                    metrics.record_ai_autocontinue_call(final_provider, final_model)
                                except Exception:
                                    pass
                                # 非流式续写调用也使用 nonstream_url
                                addition = await self._auto_continue_completion(client, nonstream_url if 'nonstream_url' in locals() else current_url, headers_json, limiter, request_data, full_content, missing_sections, round_ix=rounds)
                                addition = self._merge_dedup_addition(full_content, addition)
                                if addition:
                                    full_content += addition
                                    for c in self._chunk_text(addition):
                                        chunks_sent += 1
                                        yield json.dumps({
                                            "stock_code": stock_code,
                                            "ai_analysis_chunk": c,
                                            "event": "ai_delta",
                                            "delta": c,
                                            "status": "analyzing"
                                        })
                                missing_sections = self._detect_missing_sections(full_content)
                                rounds += 1
                    else:
                        missing_sections = []

                    # 完结事件
                    if full_content:
                        recommendation = self._extract_recommendation(full_content)
                        score = self._calculate_analysis_score(full_content, technical_summary)
                    else:
                        recommendation = "观望"
                        score = self._calculate_analysis_score("", technical_summary)
                    yield json.dumps({
                        "stock_code": stock_code,
                        "status": "completed",
                        "score": score,
                        "recommendation": recommendation,
                        "char_count": len(full_content),
                        "sections_missing": missing_sections,
                        "sections_ok": len(missing_sections) == 0,
                        "finish_reason": saw_finish_reason or outcome,
                        "chunks_sent": chunks_sent,
                        "ai_provider": final_provider,
                        "ai_model": final_model,
                        "fallback_reason": fallback_reason if not buffer else None
                    })
                    try:
                        elapsed = time.perf_counter() - start_time
                        metrics.observe_ai_stream_duration(elapsed, model=self.API_MODEL, outcome=outcome)
                        metrics.record_ai_output_chars(final_provider, final_model, len(full_content))
                    except Exception:
                        pass
                else:
                    # 非流式响应处理（带限流、重试、降级）
                    max_attempts = 3
                    attempt = 0
                    analysis_text = ""
                    temp = 0.7
                    top_p = None
                    max_toks = self.API_MAX_TOKENS
                    while attempt < max_attempts and not analysis_text:
                        # 构造非流式载荷（根据 provider 映射参数）
                        if (self.AI_PROVIDER or "") == "gemini":
                            payload = {
                                "contents": [
                                    {
                                        "role": "user",
                                        "parts": [{"text": prompt if attempt == 0 else f"{prompt}\n请简洁回答（不超过180字）。"}],
                                    }
                                ],
                                "generationConfig": {
                                    "temperature": temp,
                                    "maxOutputTokens": max_toks,
                                },
                            }
                        else:
                            payload = dict(request_data)
                            payload["stream"] = False
                            payload["temperature"] = temp
                            if top_p is not None:
                                payload["top_p"] = top_p
                            payload["max_tokens"] = max_toks
                            if attempt >= 1:
                                try:
                                    msgs = list(payload.get("messages", []))
                                    if msgs and isinstance(msgs[0], dict):
                                        msgs[0]["content"] = str(msgs[0]["content"]) + "\n请简洁回答（不超过180字）。"
                                        payload["messages"] = msgs
                                except Exception:
                                    pass
                        await limiter.wait_for_slot()
                        try:
                            response = await client.post(api_url, json=payload, headers=headers_json)
                            if response.status_code == 200:
                                response_data = response.json()
                                choices_list = response_data.get("choices", [])
                                if isinstance(choices_list, list) and len(choices_list) > 0:
                                    first_choice = choices_list[0] or {}
                                    message = first_choice.get("message", {})
                                    if isinstance(message, dict) and message:
                                        analysis_text = message.get("content", "") or ""
                                    else:
                                        analysis_text = first_choice.get("text", "") or ""
                                else:
                                    # 兼容其他返回格式（如 Gemini）
                                    analysis_text = response_data.get("content", "") or response_data.get("text", "") or ""
                                    if not analysis_text and isinstance(response_data, dict):
                                        cands = response_data.get("candidates") or []
                                        if isinstance(cands, list) and cands:
                                            cand0 = cands[0] or {}
                                            content_obj = cand0.get("content") or {}
                                            parts = content_obj.get("parts") or []
                                            if isinstance(parts, list) and parts:
                                                analysis_text = parts[0].get("text") or ""
                                await limiter.on_success()
                                limiter.release()
                                if analysis_text:
                                    break
                            elif response.status_code in (429, 500, 502, 503, 504):
                                delay = await limiter.on_transient_error(response.status_code, response.headers.get("Retry-After") if hasattr(response, "headers") else None)
                                try:
                                    metrics.record_ai_retry(self.PROVIDER, self.API_MODEL, reason=str(response.status_code))
                                except Exception:
                                    pass
                                limiter.release()
                                await asyncio.sleep(delay)
                                attempt += 1
                                # 自适应降级
                                max_toks = max(128, int(max_toks * 0.7))
                                temp = max(0.2, temp * 0.8)
                                top_p = 0.8 if top_p is None else max(0.6, top_p * 0.9)
                                continue
                            else:
                                limiter.release()
                                logger.error(f"[{correlation_id}] AI API请求失败: {response.status_code}")
                                break
                        except httpx.RequestError as e:
                            try:
                                metrics.record_ai_retry(self.PROVIDER, self.API_MODEL, reason="network")
                            except Exception:
                                pass
                            limiter.release()
                            logger.error(f"[{correlation_id}] AI API请求错误: {str(e)}")
                            await asyncio.sleep(RateLimiter.compute_backoff(attempt))
                            attempt += 1

                    # 非流式也做完整性控制（合并返回）
                    if analysis_text:
                        # 补全至达标
                        full_text = analysis_text
                        missing_sections = self._detect_missing_sections(full_text)
                        rounds = 0
                        while rounds < 2 and (len(full_text) < self.MIN_OUTPUT_CHARS or missing_sections):
                            try:
                                metrics.record_ai_autocontinue_call(self.PROVIDER, self.API_MODEL)
                            except Exception:
                                pass
                            addition = await self._auto_continue_completion(client, api_url, headers_json, limiter, request_data, full_text, missing_sections, round_ix=rounds)
                            addition = self._merge_dedup_addition(full_text, addition)
                            if addition:
                                full_text += addition
                            missing_sections = self._detect_missing_sections(full_text)
                            rounds += 1
                        try:
                            self.cache.set(cache_key, {"analysis_text": full_text}, ttl_seconds=self.cache_ttl)
                        except Exception:
                            pass
                        recommendation = self._extract_recommendation(full_text)
                        score = self._calculate_analysis_score(full_text, technical_summary)
                        yield json.dumps({
                            "stock_code": stock_code,
                            "status": "completed",
                            "analysis": full_text,
                            "score": score,
                            "recommendation": recommendation,
                            "rsi": rsi,
                            "price": price,
                            "price_change": price_change,
                            "ma_trend": ma_trend,
                            "macd_signal": macd_signal_type,
                            "volume_status": volume_status,
                            "analysis_date": analysis_date,
                            "char_count": len(full_text),
                            "sections_missing": missing_sections,
                            "sections_ok": len(missing_sections) == 0,
                            "finish_reason": "nonstream"
                        })
                    else:
                        note = generate_placeholder(stock_code, technical_summary)
                        try:
                            metrics.record_ai_fallback(self.PROVIDER, self.API_MODEL, reason="placeholder")
                        except Exception:
                            pass
                        yield json.dumps({
                            "stock_code": stock_code,
                            "status": "completed",
                            "analysis": note,
                            "score": self._calculate_analysis_score(note, technical_summary),
                            "recommendation": self._extract_recommendation(note),
                            "rsi": rsi,
                            "price": price,
                            "price_change": price_change,
                            "ma_trend": ma_trend,
                            "macd_signal": macd_signal_type,
                            "volume_status": volume_status,
                            "analysis_date": analysis_date,
                            "char_count": len(note),
                            "sections_missing": self._detect_missing_sections(note),
                            "sections_ok": False,
                            "finish_reason": "degraded"
                        })

        except Exception as e:
            logger.error(f"AI分析出错: {str(e)}", exc_info=True)
            yield json.dumps({
                "stock_code": stock_code,
                "error": f"分析出错: {str(e)}",
                "status": "error"
            })

    def _extract_recommendation(self, analysis_text: str) -> str:
        """从分析文本中提取投资建议"""
        # 查找投资建议部分
        investment_advice_pattern = r"##\s*投资建议\s*\n(.*?)(?:\n##|\Z)"
        match = re.search(investment_advice_pattern, analysis_text, re.DOTALL)

        if match:
            advice_text = match.group(1).strip()

            # 提取关键建议
            if "买入" in advice_text or "增持" in advice_text:
                return "买入"
            elif "卖出" in advice_text or "减持" in advice_text:
                return "卖出"
            elif "持有" in advice_text:
                return "持有"
            else:
                return "观望"

        return "观望"  # 默认建议

    def _calculate_analysis_score(self, analysis_text: str, technical_summary: dict) -> int:
        """计算分析评分"""
        score = 50  # 基础分数

        # 根据技术指标调整分数
        if technical_summary['trend'] == 'upward':
            score += 10
        else:
            score -= 10

        if technical_summary['volume_trend'] == 'increasing':
            score += 5
        else:
            score -= 5

        rsi = technical_summary['rsi_level']
        if rsi < 30:  # 超卖
            score += 15
        elif rsi > 70:  # 超买
            score -= 15

        # 根据分析文本中的关键词调整分数
        if "强烈买入" in analysis_text or "显著上涨" in analysis_text:
            score += 20
        elif "买入" in analysis_text or "看涨" in analysis_text:
            score += 10
        elif "强烈卖出" in analysis_text or "显著下跌" in analysis_text:
            score -= 20
        elif "卖出" in analysis_text or "看跌" in analysis_text:
            score -= 10

        # 确保分数在0-100范围内
        return max(0, min(100, score))

    # ---- 新增：输出完整性与分块工具 ----
    def _split_sentences(self, text: str):
        try:
            parts = re.split(r'(?<=[。！？!\?；;\n])', text)
            return [p for p in (parts or []) if p and p.strip()]
        except Exception:
            return [text]

    def _drain_chunks_from_buffer(self, buf: str, max_bytes: int = None, min_sentences: int = None):
        max_b = int(max_bytes or self.CHUNK_MAX_BYTES)
        min_s = int(min_sentences or self.CHUNK_MIN_SENTENCES)
        sentences = self._split_sentences(buf)
        if not sentences:
            return [], buf
        # keep last partial sentence as remainder if buffer doesn't end with delimiter
        remainder = ""
        if not (buf.endswith("\n") or re.search(r'[。！？!\?；;]$', buf)):
            remainder = sentences.pop() if sentences else ""
        chunks = []
        cur = ""
        cur_sents = 0
        for s in sentences:
            eb = len((cur + s).encode('utf-8'))
            if cur and (eb > max_b) and (cur_sents >= min_s):
                chunks.append(cur)
                cur = ""
                cur_sents = 0
            cur += s
            cur_sents += 1
        if cur:
            if cur_sents >= min_s or len(cur.encode('utf-8')) >= max_b:
                chunks.append(cur)
            else:
                remainder = cur + remainder
        return chunks, remainder

    def _chunk_text(self, text: str, max_bytes: int = None, min_sentences: int = None):
        chunks, rem = self._drain_chunks_from_buffer(text, max_bytes=max_bytes, min_sentences=min_sentences)
        if rem and rem.strip():
            chunks.append(rem)
        return chunks

    def _merge_dedup_addition(self, base: str, addition: str) -> str:
        if not base:
            return addition or ""
        add = addition or ""
        try:
            max_overlap = min(120, len(add))
            for i in range(max_overlap, 10, -1):
                if base.endswith(add[:i]):
                    return add[i:]
            return add
        except Exception:
            return add

    def _detect_missing_sections(self, text: str):
        missing = []
        body = text or ""
        try:
            patterns = {
                "概览": r"(?:^|\n)\s*(?:[#【 ]{0,2})?(?:概览|总体|总览|概述|概况)",
                "主要信号": r"(?:^|\n)\s*(?:[#【 ]{0,2})?(?:主要信号|关键信号|亮点|要点)",
                "技术面": r"(?:^|\n)\s*(?:[#【 ]{0,2})?(?:技术面|技术分析|技术指标)",
                "基本面": r"(?:^|\n)\s*(?:[#【 ]{0,2})?(?:基本面|基本分析|基本情况|财务|估值)",
                "风险": r"(?:^|\n)\s*(?:[#【 ]{0,2})?(?:风险|不利因素|注意事项)",
                "操作建议": r"(?:^|\n)\s*(?:[#【 ]{0,2})?(?:操作建议|投资建议|策略|建议)"
            }
            for sec in self.REQUIRED_SECTIONS:
                pat = patterns.get(sec)
                if not pat or not re.search(pat, body, flags=re.IGNORECASE):
                    missing.append(sec)
        except Exception:
            pass
        return missing

    async def _auto_continue_completion(self, client, api_url: str, headers: dict, limiter, base_payload: dict, current_text: str, missing_sections, round_ix: int = 0):
        try:
            contend = "、".join(missing_sections or [])
            prompt_suffix = (
                "\n请基于已生成的内容，仅补全缺失段落并延续风格，避免重复，中文回答。"
                f"缺失段落: {contend if contend else '无'}。"
                "注意：严格输出六段结构（概览/主要信号/技术面/基本面/风险/操作建议），如果已有某段，请勿重写。"
                "直接给出新增内容。"
            )
            payload = dict(base_payload)
            payload["stream"] = False
            payload["temperature"] = 0.2
            payload["top_p"] = 0.8
            payload["max_tokens"] = 256
            msgs = list(payload.get("messages", []))
            last = (current_text or "")[-2000:]
            msgs.append({
                "role": "user",
                "content": f"已生成内容（节选）：\n{last}\n{prompt_suffix}"
            })
            payload["messages"] = msgs

            await limiter.wait_for_slot()
            try:
                resp = await client.post(api_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    ch = (data.get("choices") or [{}])[0]
                    msg = ch.get("message") or {}
                    cont = msg.get("content") or ch.get("text") or data.get("content") or ""
                    await limiter.on_success()
                    return cont or ""
                elif resp.status_code in (429, 500, 502, 503, 504):
                    delay = await limiter.on_transient_error(resp.status_code, resp.headers.get("Retry-After") if hasattr(resp, "headers") else None)
                    await asyncio.sleep(delay)
                    return ""
                else:
                    return ""
            except httpx.RequestError:
                await asyncio.sleep(0.25)
                return ""
            finally:
                limiter.release()
        except Exception:
            return ""
