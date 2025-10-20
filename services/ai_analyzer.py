import pandas as pd
import os
import json
import httpx
import re
import asyncio
import time
from typing import AsyncGenerator
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.api_utils import APIUtils
from utils import metrics
from datetime import datetime

# 获取日志器
logger = get_logger()

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
        
        # 设置API配置
        self.API_URL = custom_api_url or os.getenv('API_URL')
        self.API_KEY = custom_api_key or os.getenv('API_KEY')
        self.API_MODEL = custom_api_model or os.getenv('API_MODEL', 'gpt-3.5-turbo')
        self.API_TIMEOUT = int(custom_api_timeout or os.getenv('API_TIMEOUT', 60))
        self.API_MAX_TOKENS = int(os.getenv('API_MAX_TOKENS', '1024'))
        
        logger.debug(f"初始化AIAnalyzer: API_URL={self.API_URL}, API_MODEL={self.API_MODEL}, API_KEY={'已提供' if self.API_KEY else '未提供'}, API_TIMEOUT={self.API_TIMEOUT}")
    
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
            # 最近14天的股票数据记录
            recent_data = df.tail(14).to_dict('records')
            
            # 包含trend, volatility, volume_trend, rsi_level的字典
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
            
            # 格式化API URL
            api_url = APIUtils.format_api_url(self.API_URL)
            
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
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.API_KEY}"
            }
            
            # 获取当前日期作为分析日期
            analysis_date = datetime.now().strftime("%Y-%m-%d")
            
            # 异步请求API
            timeout = httpx.Timeout(read=self.API_TIMEOUT, connect=30.0, write=self.API_TIMEOUT, pool=self.API_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                # 记录请求
                logger.debug(f"发送AI请求: URL={api_url}, MODEL={self.API_MODEL}, STREAM={stream}")
                
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
                    # 流式响应处理
                    first_chunk_snapshot = None
                    last_chunk_snapshot = None
                    chunk_count = 0
                    buffer = ""
                    
                    try:
                        async with client.stream("POST", api_url, json=request_data, headers=headers) as response:
                            if response.status_code != 200:
                                # 记录错误并尝试降级
                                try:
                                    error_text = await response.aread()
                                    error_data = json.loads(error_text)
                                    error_message = error_data.get('error', {}).get('message', '未知错误')
                                except Exception:
                                    error_message = f"HTTP {response.status_code}"
                                logger.error(f"AI API流式请求失败: {response.status_code} - {error_message}")
                                chunk_count = 0
                            else:
                                async for chunk in response.aiter_text():
                                    if not chunk:
                                        continue
                                    # 分割多行响应（处理某些API可能在一个chunk中返回多行）
                                    lines = chunk.strip().split('\n')
                                    for line in lines:
                                        line = line.strip()
                                        if not line:
                                            continue
                                        # 处理以data:开头的行
                                        if line.startswith("data: "):
                                            line = line[6:]
                                        if line == "[DONE]":
                                            logger.debug("收到流结束标记 [DONE]")
                                            continue
                                        try:
                                            # 处理特殊错误情况
                                            if "error" in line.lower() and line.strip().startswith('{'):
                                                try:
                                                    _err = json.loads(line)
                                                    err_msg = _err.get("error", line)
                                                except Exception:
                                                    err_msg = line
                                                logger.error(f"流式响应中收到错误: {err_msg}")
                                                continue
                                            # 尝试解析JSON
                                            chunk_data = json.loads(line)
                                            choices_list = chunk_data.get("choices", [])
                                            if not isinstance(choices_list, list) or len(choices_list) == 0:
                                                logger.debug("流式响应中choices为空或格式异常，跳过该块")
                                                continue
                                            first_choice = choices_list[0] or {}
                                            finish_reason = first_choice.get("finish_reason")
                                            if finish_reason in ("stop", "length"):
                                                logger.debug(f"收到finish_reason={finish_reason}")
                                                continue
                                            delta = first_choice.get("delta", {}) or {}
                                            if not delta:
                                                logger.debug("收到空的delta对象，跳过")
                                                continue
                                            content = delta.get("content")
                                            if content:
                                                if first_chunk_snapshot is None:
                                                    first_chunk_snapshot = content[:100]
                                                last_chunk_snapshot = content[-100:]
                                                chunk_count += 1
                                                buffer += content
                                                yield json.dumps({
                                                    "stock_code": stock_code,
                                                    "ai_analysis_chunk": content,
                                                    "status": "analyzing"
                                                })
                                        except json.JSONDecodeError:
                                            logger.error(f"JSON解析错误，块内容: {line}")
                                            if "streaming failed after retries" in line.lower():
                                                logger.error("检测到流式传输失败")
                                                chunk_count = 0
                                                break
                                            continue
                    except httpx.RequestError as e:
                        logger.error(f"流式请求发生网络错误: {str(e)}")
                        chunk_count = 0
                    
                    logger.info(f"AI流式处理完成，共收到 {chunk_count} 个内容片段，总长度: {len(buffer)}")
                    if first_chunk_snapshot is not None:
                        logger.debug(f"首帧快照: {first_chunk_snapshot}")
                    if last_chunk_snapshot is not None:
                        logger.debug(f"末帧快照: {last_chunk_snapshot}")
                    
                    # 如果buffer不为空且不以换行符结束，发送一个换行符
                    if buffer and not buffer.endswith('\n'):
                        yield json.dumps({
                            "stock_code": stock_code,
                            "ai_analysis_chunk": "\n",
                            "status": "analyzing"
                        })
                    
                    # 若无任何片段，触发一次非流式补救请求
                    if chunk_count == 0 or (buffer.strip() == ""):
                        logger.warning("AI流式分析返回0片段，触发非流式降级补救")
                        try:
                            metrics.record_ai_stream_zero_chunks(self.API_MODEL, reason="zero_chunks")
                            metrics.record_ai_stream_fallback(self.API_MODEL, reason="zero_chunks")
                        except Exception:
                            pass
                        # 补救请求（最多重试2次，指数退避）
                        analysis_text = ""
                        attempt = 0
                        backoff = 0.5
                        nonstream_payload = dict(request_data)
                        nonstream_payload["stream"] = False
                        while attempt < 2 and not analysis_text:
                            try:
                                resp = await client.post(api_url, json=nonstream_payload, headers=headers)
                                if resp.status_code == 200:
                                    resp_data = resp.json()
                                    choices_list = resp_data.get("choices", [])
                                    if isinstance(choices_list, list) and len(choices_list) > 0:
                                        first_choice = choices_list[0] or {}
                                        message = first_choice.get("message", {})
                                        if isinstance(message, dict) and message:
                                            analysis_text = message.get("content", "") or ""
                                        else:
                                            analysis_text = first_choice.get("text", "") or ""
                                    else:
                                        analysis_text = resp_data.get("content", "") or resp_data.get("text", "") or ""
                                else:
                                    logger.warning(f"非流式补救请求失败: HTTP {resp.status_code}")
                            except httpx.RequestError as e:
                                logger.warning(f"非流式补救请求网络错误: {str(e)}")
                            if not analysis_text:
                                await asyncio.sleep(backoff)
                                backoff *= 2
                                attempt += 1
                        if analysis_text:
                            # 回放为一个整块，以保持前端向后兼容（仍然是chunk事件）
                            yield json.dumps({
                                "stock_code": stock_code,
                                "ai_analysis_chunk": analysis_text,
                                "status": "analyzing"
                            })
                            full_content = analysis_text
                            outcome = "fallback"
                        else:
                            # 补救失败，输出一个简短说明
                            note = "暂无可用分析内容(降级失败)"
                            yield json.dumps({
                                "stock_code": stock_code,
                                "ai_analysis_chunk": note,
                                "status": "analyzing"
                            })
                            full_content = note
                            outcome = "error"
                    else:
                        full_content = buffer
                        outcome = "ok"
                    
                    # 计算评分与建议并结束
                    recommendation = self._extract_recommendation(full_content)
                    score = self._calculate_analysis_score(full_content, technical_summary)
                    yield json.dumps({
                        "stock_code": stock_code,
                        "status": "completed",
                        "score": score,
                        "recommendation": recommendation
                    })
                    try:
                        elapsed = time.perf_counter() - start_time
                        metrics.observe_ai_stream_duration(elapsed, model=self.API_MODEL, outcome=outcome)
                    except Exception:
                        pass
                else:
                    # 非流式响应处理
                    try:
                        response = await client.post(api_url, json=request_data, headers=headers)
                    except httpx.RequestError as e:
                        logger.error(f"AI API请求错误: {str(e)}")
                        yield json.dumps({
                            "stock_code": stock_code,
                            "error": f"API请求错误: {str(e)}",
                            "status": "error"
                        })
                        return
                    
                    if response.status_code != 200:
                        try:
                            error_data = response.json()
                            error_message = error_data.get('error', {}).get('message', '未知错误')
                        except Exception:
                            error_message = f"HTTP {response.status_code}"
                        logger.error(f"AI API请求失败: {response.status_code} - {error_message}")
                        yield json.dumps({
                            "stock_code": stock_code,
                            "error": f"API请求失败: {error_message}",
                            "status": "error"
                        })
                        return
                    
                    response_data = response.json()
                    analysis_text = ""
                    choices_list = response_data.get("choices", [])
                    if isinstance(choices_list, list) and len(choices_list) > 0:
                        first_choice = choices_list[0] or {}
                        # 优先从chat格式中获取内容
                        message = first_choice.get("message", {})
                        if isinstance(message, dict) and message:
                            analysis_text = message.get("content", "") or ""
                        else:
                            # 兼容部分提供方的text字段
                            analysis_text = first_choice.get("text", "") or ""
                    else:
                        # 顶层备用字段
                        analysis_text = response_data.get("content", "") or response_data.get("text", "") or ""
                    
                    # 尝试从分析内容中提取投资建议
                    recommendation = self._extract_recommendation(analysis_text)
                    
                    # 计算分析评分
                    score = self._calculate_analysis_score(analysis_text, technical_summary)
                    
                    # 发送完整的分析结果
                    yield json.dumps({
                        "stock_code": stock_code,
                        "status": "completed",
                        "analysis": analysis_text,
                        "score": score,
                        "recommendation": recommendation,
                        "rsi": rsi,
                        "price": price,
                        "price_change": price_change,
                        "ma_trend": ma_trend,
                        "macd_signal": macd_signal_type,
                        "volume_status": volume_status,
                        "analysis_date": analysis_date
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
    
    def _truncate_json_for_logging(self, json_obj, max_length=500):
        """
        截断JSON对象以便记录日志
        
        Args:
            json_obj: JSON对象
            max_length: 最大长度
            
        Returns:
            截断后的字符串
        """
        json_str = json.dumps(json_obj, ensure_ascii=False)
        if len(json_str) <= max_length:
            return json_str
        return json_str[:max_length] + "..." 