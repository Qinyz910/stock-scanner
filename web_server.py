from fastapi import FastAPI, Request, Response, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Generator
from services.stock_analyzer_service import StockAnalyzerService
from services.us_stock_service_async import USStockServiceAsync
from services.fund_service_async import FundServiceAsync
from services.factor_scoring import FactorScoringEngine, FactorDef, TransformConfig
import os
import httpx
from utils.logger import get_logger
from utils.api_utils import APIUtils
from utils.error_handlers import register_error_handlers, ErrorResponse
from utils.exceptions import (
    AppException,
    UnauthorizedError,
    ValidationError as AppValidationError,
    NotFoundError,
    UpstreamError,
    InternalError,
)
from dotenv import load_dotenv
import uvicorn
import json
import secrets
from datetime import datetime, timedelta
from jose import JWTError, jwt
from utils import metrics
from api_v2 import api_v2_router

load_dotenv()

# 获取日志器
logger = get_logger()

# JWT相关配置
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 10080  # Token过期时间一周

LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "")
print(LOGIN_PASSWORD)

# 是否需要登录
REQUIRE_LOGIN = bool(LOGIN_PASSWORD.strip())


app = FastAPI(
    title="Stock Scanner API",
    description="异步股票分析API",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Errors",
            "description": "Error response definitions for API documentation",
        }
    ]
)

# Register unified error handlers
register_error_handlers(app)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],
)

# Prometheus metrics instrumentation and endpoint
try:
    metrics.instrument_fastapi(app)
    metrics.mount_metrics_endpoint(app)
except Exception:
    pass

# Mount v2 API router (features gated within endpoints)
app.include_router(api_v2_router)


# Custom OpenAPI schema configuration for error responses
def custom_openapi():
    """Customize OpenAPI schema to include error response components."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = app.openapi()
    
    # Add component schemas for error responses
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "schemas" not in openapi_schema["components"]:
        openapi_schema["components"]["schemas"] = {}
    
    openapi_schema["components"]["schemas"]["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Error code (e.g., VALIDATION_ERROR, UNAUTHORIZED, NOT_FOUND, etc.)",
                "example": "VALIDATION_ERROR"
            },
            "message": {
                "type": "string",
                "description": "Human-readable error message",
                "example": "Request validation failed"
            },
            "details": {
                "type": "object",
                "description": "Additional error details",
                "example": {"errors": [{"field": "email", "type": "value_error", "message": "invalid email format"}]}
            },
            "traceId": {
                "type": "string",
                "description": "Trace ID for error tracking and debugging",
                "example": "550e8400-e29b-41d4-a716-446655440000"
            }
        },
        "required": ["code", "message"]
    }
    
    # Add error response definitions
    openapi_schema["components"]["schemas"]["ValidationErrorResponse"] = {
        "allOf": [
            {"$ref": "#/components/schemas/ErrorResponse"},
            {
                "type": "object",
                "properties": {
                    "code": {
                        "enum": ["VALIDATION_ERROR"]
                    }
                }
            }
        ]
    }
    
    openapi_schema["components"]["schemas"]["UnauthorizedErrorResponse"] = {
        "allOf": [
            {"$ref": "#/components/schemas/ErrorResponse"},
            {
                "type": "object",
                "properties": {
                    "code": {
                        "enum": ["UNAUTHORIZED"]
                    }
                }
            }
        ]
    }
    
    openapi_schema["components"]["schemas"]["ForbiddenErrorResponse"] = {
        "allOf": [
            {"$ref": "#/components/schemas/ErrorResponse"},
            {
                "type": "object",
                "properties": {
                    "code": {
                        "enum": ["FORBIDDEN"]
                    }
                }
            }
        ]
    }
    
    openapi_schema["components"]["schemas"]["NotFoundErrorResponse"] = {
        "allOf": [
            {"$ref": "#/components/schemas/ErrorResponse"},
            {
                "type": "object",
                "properties": {
                    "code": {
                        "enum": ["NOT_FOUND"]
                    }
                }
            }
        ]
    }
    
    openapi_schema["components"]["schemas"]["UpstreamErrorResponse"] = {
        "allOf": [
            {"$ref": "#/components/schemas/ErrorResponse"},
            {
                "type": "object",
                "properties": {
                    "code": {
                        "enum": ["UPSTREAM_ERROR"]
                    }
                }
            }
        ]
    }
    
    openapi_schema["components"]["schemas"]["InternalErrorResponse"] = {
        "allOf": [
            {"$ref": "#/components/schemas/ErrorResponse"},
            {
                "type": "object",
                "properties": {
                    "code": {
                        "enum": ["INTERNAL_ERROR"]
                    }
                }
            }
        ]
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# ---- Health: AI provider ----
AI_STARTUP_CONF = None
try:
    AI_STARTUP_CONF = APIUtils.resolve_ai_config()
    if AI_STARTUP_CONF.get('ok'):
        logger.info(f"AI 配置检测通过: provider={AI_STARTUP_CONF.get('provider')}, model={AI_STARTUP_CONF.get('model')}, base_url={AI_STARTUP_CONF.get('base_url')}")
    else:
        logger.error(f"AI 配置缺失: provider={AI_STARTUP_CONF.get('provider')}, model={AI_STARTUP_CONF.get('model')}, base_url={'set' if AI_STARTUP_CONF.get('base_url') else 'missing'}, errors={AI_STARTUP_CONF.get('errors')}")
except Exception as _ex:
    logger.exception(_ex)


@app.get("/health/ai")
async def health_ai():
    conf = APIUtils.resolve_ai_config()
    provider = conf.get('provider')
    base_url = conf.get('base_url') or ""
    model = conf.get('model') or ""
    ok = conf.get('ok')
    errors = conf.get('errors')
    try:
        stream_url = APIUtils.format_ai_url(base_url, model=model, provider=provider, stream=True)
        nonstream_url = APIUtils.format_ai_url(base_url, model=model, provider=provider, stream=False)
    except Exception:
        stream_url = APIUtils.format_api_url(base_url)
        nonstream_url = APIUtils.format_api_url(base_url)
    return {
        "provider": provider,
        "configured": bool(base_url) and bool(conf.get('api_key_present')),
        "model": model,
        "base_url": base_url,
        "endpoints": {
            "stream": stream_url,
            "non_stream": nonstream_url,
        },
        "status": "ok" if ok else "error",
        "errors": errors,
    }

# 初始化异步服务
us_stock_service = USStockServiceAsync()
fund_service = FundServiceAsync()
# 因子评分引擎
factor_engine = FactorScoringEngine()

# 定义请求和响应模型
class AnalyzeRequest(BaseModel):
    stock_codes: List[str]
    market_type: str = "A"
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    api_model: Optional[str] = None
    api_timeout: Optional[str] = None

class TestAPIRequest(BaseModel):
    api_url: str
    api_key: str
    api_model: Optional[str] = None
    api_timeout: Optional[int] = 10

class TransformModel(BaseModel):
    winsorize_lower: float = 0.05
    winsorize_upper: float = 0.95
    standardize: bool = True
    fillna: Optional[float] = None
    industry_neutral: bool = False

class FactorModel(BaseModel):
    id: str
    name: Optional[str] = None
    weight: float = 1.0
    params: Dict[str, Any] = Field(default_factory=dict)
    transform: TransformModel = Field(default_factory=TransformModel)

class ScoresRequest(BaseModel):
    symbols: List[str]
    market_type: str = "A"
    factors: List[FactorModel]
    window: int = 20
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    industries: Optional[Dict[str, str]] = None
    page: int = 1
    page_size: int = 100

class LoginRequest(BaseModel):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# 自定义依赖项，在REQUIRE_LOGIN=False时不要求token
class OptionalOAuth2PasswordBearer(OAuth2PasswordBearer):
    async def __call__(self, request: Request) -> Optional[str]:
        if not REQUIRE_LOGIN:
            return None
        try:
            return await super().__call__(request)
        except HTTPException:
            if not REQUIRE_LOGIN:
                return None
            raise

# 使用自定义的依赖项
optional_oauth2_scheme = OptionalOAuth2PasswordBearer(tokenUrl="login")

# 创建访问令牌
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 验证令牌
async def verify_token(token: Optional[str] = Depends(optional_oauth2_scheme)):
    # 如果未设置密码，则不需要验证
    if not REQUIRE_LOGIN:
        return "guest"
        
    # 如果没有token且不需要登录，返回guest
    if token is None and not REQUIRE_LOGIN:
        return "guest"
        
    # 如果需要登录但没有token，抛出异常
    if token is None:
        raise UnauthorizedError("Invalid authentication credentials")
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise UnauthorizedError("Invalid authentication token")
        return username
    except JWTError as e:
        raise UnauthorizedError(
            "Invalid or expired authentication token",
            original_error=e
        )

# 用户登录接口
@app.post("/api/login")
async def login(request: LoginRequest):
    """用户登录接口"""
    # 如果未设置密码，表示不需要登录
    if not REQUIRE_LOGIN:
        access_token = create_access_token(data={"sub": "guest"})
        return {"access_token": access_token, "token_type": "bearer"}
        
    if request.password != LOGIN_PASSWORD:
        logger.warning("登录失败：密码错误")
        raise UnauthorizedError("Invalid password")
    
    # 创建访问令牌
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "user"}, expires_delta=access_token_expires
    )
    logger.info("用户登录成功")
    return {"access_token": access_token, "token_type": "bearer"}

# 检查用户认证状态
@app.get("/api/check_auth")
async def check_auth(username: str = Depends(verify_token)):
    """检查用户认证状态"""
    return {"authenticated": True, "username": username}

# 获取系统配置
@app.get("/api/config")
async def get_config():
    """返回系统配置信息"""
    config = {
        'announcement': os.getenv('ANNOUNCEMENT_TEXT') or '',
        'default_api_url': os.getenv('API_URL', ''),
        'default_api_model': os.getenv('API_MODEL', ''),
        'default_api_timeout': os.getenv('API_TIMEOUT', '60')
    }
    return config

# AI分析股票
@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest, username: str = Depends(verify_token)):
    try:
        logger.info("开始处理分析请求")
        stock_codes = request.stock_codes
        market_type = request.market_type
        
        # 后端再次去重，确保安全
        original_count = len(stock_codes)
        stock_codes = list(dict.fromkeys(stock_codes))  # 保持原有顺序的去重方法
        if len(stock_codes) < original_count:
            logger.info(f"后端去重: 从{original_count}个代码中移除了{original_count - len(stock_codes)}个重复项")
        
        logger.debug(f"接收到分析请求: stock_codes={stock_codes}, market_type={market_type}")
        
        # 获取自定义API配置
        custom_api_url = request.api_url
        custom_api_key = request.api_key
        custom_api_model = request.api_model
        custom_api_timeout = request.api_timeout
        
        logger.debug(f"自定义API配置: URL={custom_api_url}, 模型={custom_api_model}, API Key={'已提供' if custom_api_key else '未提供'}, Timeout={custom_api_timeout}")
        
        # 创建新的分析器实例，使用自定义配置
        custom_analyzer = StockAnalyzerService(
            custom_api_url=custom_api_url,
            custom_api_key=custom_api_key,
            custom_api_model=custom_api_model,
            custom_api_timeout=custom_api_timeout
        )
        
        if not stock_codes:
            logger.warning("未提供股票代码")
            raise AppValidationError("Stock codes are required")
        
        # 定义流式生成器
        async def generate_stream():
            if len(stock_codes) == 1:
                # 单个股票分析流式处理
                stock_code = stock_codes[0].strip()
                logger.info(f"开始单股流式分析: {stock_code}")
                
                stock_code_json = json.dumps(stock_code)
                init_message = f'{{"stream_type": "single", "stock_code": {stock_code_json}}}\n'
                yield init_message
                
                logger.debug(f"开始处理股票 {stock_code} 的流式响应")
                chunk_count = 0
                
                # 使用异步生成器
                async for chunk in custom_analyzer.analyze_stock(stock_code, market_type, stream=True):
                    chunk_count += 1
                    yield chunk + '\n'
                
                logger.info(f"股票 {stock_code} 流式分析完成，共发送 {chunk_count} 个块")
            else:
                # 批量分析流式处理
                logger.info(f"开始批量流式分析: {stock_codes}")
                
                stock_codes_json = json.dumps(stock_codes)
                init_message = f'{{"stream_type": "batch", "stock_codes": {stock_codes_json}}}\n'
                yield init_message
                
                logger.debug(f"开始处理批量股票的流式响应")
                chunk_count = 0
                
                # 使用异步生成器
                async for chunk in custom_analyzer.scan_stocks(
                    [code.strip() for code in stock_codes], 
                    min_score=0, 
                    market_type=market_type,
                    stream=True
                ):
                    chunk_count += 1
                    yield chunk + '\n'
                
                logger.info(f"批量流式分析完成，共发送 {chunk_count} 个块")
        
        logger.info("成功创建流式响应生成器")
        return StreamingResponse(generate_stream(), media_type='application/json')
            
    except AppValidationError:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred during analysis",
            original_error=e
        )

# 因子评分 API
@app.post("/api/scores")
async def compute_scores(request: ScoresRequest, username: str = Depends(verify_token)):
    try:
        if not request.symbols:
            raise AppValidationError("Stock symbols are required")
        symbols = list(dict.fromkeys([s.strip() for s in request.symbols]))
        if not request.factors:
            raise AppValidationError("Factors list is required")

        factor_defs: List[FactorDef] = []
        for f in request.factors:
            tcfg = TransformConfig(
                winsorize_lower=f.transform.winsorize_lower,
                winsorize_upper=f.transform.winsorize_upper,
                standardize=f.transform.standardize,
                fillna=f.transform.fillna,
                industry_neutral=f.transform.industry_neutral,
            )
            factor_defs.append(
                FactorDef(
                    id=f.id,
                    name=f.name,
                    weight=f.weight,
                    params=dict(f.params or {}),
                    transform=tcfg,
                )
            )

        result = await factor_engine.score(
            symbols=symbols,
            factors=factor_defs,
            market_type=request.market_type,
            window=request.window,
            start_date=request.start_date,
            end_date=request.end_date,
            industries=request.industries,
            page=request.page,
            page_size=request.page_size,
        )
        return result
    except AppException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred while computing factor scores",
            original_error=e
        )

# 搜索美股代码
@app.get("/api/search_us_stocks")
async def search_us_stocks(keyword: str = "", username: str = Depends(verify_token)):
    try:
        if not keyword:
            raise AppValidationError("Search keyword is required")
        
        # 直接使用异步服务的异步方法
        results = await us_stock_service.search_us_stocks(keyword)
        return {"results": results}
        
    except AppException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred while searching US stocks",
            original_error=e
        )

# 搜索基金代码
@app.get("/api/search_funds")
async def search_funds(keyword: str = "", market_type: str = "", username: str = Depends(verify_token)):
    try:
        if not keyword:
            raise AppValidationError("Search keyword is required")
        
        # 直接使用异步服务的异步方法
        results = await fund_service.search_funds(keyword, market_type)
        return {"results": results}
        
    except AppException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred while searching funds",
            original_error=e
        )

# 获取美股详情
@app.get("/api/us_stock_detail/{symbol}")
async def get_us_stock_detail(symbol: str, username: str = Depends(verify_token)):
    try:
        if not symbol:
            raise AppValidationError("Stock symbol is required")
        
        # 使用异步服务获取详情
        detail = await us_stock_service.get_us_stock_detail(symbol)
        return detail
        
    except AppException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred while fetching US stock details",
            original_error=e
        )

# 获取基金详情
@app.get("/api/fund_detail/{symbol}")
async def get_fund_detail(symbol: str, market_type: str = "ETF", username: str = Depends(verify_token)):
    try:
        if not symbol:
            raise AppValidationError("Fund symbol is required")
        
        # 使用异步服务获取详情
        detail = await fund_service.get_fund_detail(symbol, market_type)
        return detail
        
    except AppException:
        raise
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred while fetching fund details",
            original_error=e
        )

# 测试API连接
@app.post("/api/test_api_connection")
async def test_api_connection(request: TestAPIRequest, username: str = Depends(verify_token)):
    """测试API连接"""
    try:
        logger.info("开始测试API连接")
        api_url = request.api_url
        api_key = request.api_key
        api_model = request.api_model
        api_timeout = request.api_timeout
        
        logger.debug(f"测试API连接: URL={api_url}, 模型={api_model}, API Key={'已提供' if api_key else '未提供'}, Timeout={api_timeout}")
        
        if not api_url:
            logger.warning("未提供API URL")
            raise AppValidationError("API URL is required")

        if not api_key:
            logger.warning("未提供API Key")
            raise AppValidationError("API Key is required")
            
        # 构建API URL（根据 AI_PROVIDER 选择端点）
        provider = (os.getenv("AI_PROVIDER", "") or "").lower().strip() or "newapi"
        test_url = APIUtils.format_ai_url(api_url, model=(api_model or ""), provider=provider, stream=False)
        logger.debug(f"完整API测试URL: {test_url}")
        
        # 使用异步HTTP客户端发送测试请求
        async with httpx.AsyncClient(timeout=float(api_timeout)) as client:
            # 构造请求头
            if provider == "gemini":
                headers = {
                    "x-goog-api-key": f"{api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": "Hello, this is a test message. Please respond with 'API connection successful'."}],
                        }
                    ],
                    "generationConfig": {"maxOutputTokens": 32},
                }
            else:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": api_model or "",
                    "messages": [
                        {"role": "user", "content": "Hello, this is a test message. Please respond with 'API connection successful'."}
                    ],
                    "max_tokens": 20,
                }
            response = await client.post(
                test_url,
                headers=headers,
                json=payload,
            )
        
        # 检查响应
        if response.status_code == 200:
            logger.info(f"API 连接测试成功: {response.status_code}")
            return {"success": True, "message": "API connection successful"}
        else:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            logger.warning(f"API连接测试失败: {response.status_code} - {error_message}")
            raise UpstreamError(
                f"API connection test failed: {error_message}",
                details={"status_code": response.status_code}
            )
            
    except AppValidationError:
        raise
    except httpx.RequestError as e:
        logger.error(f"API 连接请求错误: {str(e)}")
        raise UpstreamError(
            "API connection request failed",
            original_error=e
        )
    except Exception as e:
        logger.exception(e)
        raise InternalError(
            "Error occurred while testing API connection",
            original_error=e
        )

# 检查是否需要登录
@app.get("/api/need_login")
async def need_login():
    """检查是否需要登录"""
    return {"require_login": REQUIRE_LOGIN}

# 设置静态文件
frontend_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
if os.path.exists(frontend_dist):
    # 直接挂载整个dist目录
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
    logger.info(f"前端构建目录挂载成功: {frontend_dist}")
else:
    logger.warning("前端构建目录不存在，仅API功能可用")


if __name__ == '__main__':
    uvicorn.run("web_server:app", host="0.0.0.0", port=8888, reload=True)