import os
from typing import Dict, Any


class APIUtils:
    @staticmethod
    def format_api_url(base_url: str) -> str:
        """
        格式化 API URL（向后兼容的 OpenAI/newapi 兼容路径）

        规则：
        - / 结尾：直接追加 chat/completions（保留已有版本前缀）
        - # 结尾：强制使用输入地址（移除 #）
        - 其他：默认使用 /v1/chat/completions
        """
        if not base_url:
            return ""

        if base_url.endswith('/'):
            return f"{base_url}chat/completions"
        elif base_url.endswith('#'):
            return base_url.replace('#', '')
        else:
            return f"{base_url}/v1/chat/completions"

    @staticmethod
    def format_ai_url(base_url: str, model: str = "", provider: str = "", stream: bool = True) -> str:
        """
        根据 provider/model/stream 构造对应请求地址。
        - provider == 'gemini': 使用 v1beta 官方接口
          - stream: /v1beta/models/{model}:streamGenerateContent
          - non-stream: /v1beta/models/{model}:generateContent
          - 注：API 密钥传递方式由调用方决定（query 参数 key= 或 x-goog-api-key 头部），此处不拼接密钥
        - 其他: 退化为 OpenAI/newapi 兼容的 chat/completions
        """
        p = (provider or "").lower().strip()
        if p == "gemini":
            base = (base_url or "").rstrip("/")
            suffix = ":streamGenerateContent" if stream else ":generateContent"
            if not model:
                return f"{base}/v1beta/models{suffix}"
            return f"{base}/v1beta/models/{model}{suffix}"
        # 默认 OpenAI 兼容
        return APIUtils.format_api_url(base_url)

    @staticmethod
    def resolve_ai_config() -> Dict[str, Any]:
        """
        统一解析 AI Provider 与模型/地址/密钥。

        变量解析规则：
        - provider: 读取 AI_PROVIDER（默认 newapi），统一为小写
        - newapi/openai/deepseek:
            - model: 优先 AI_MODEL，其次 API_MODEL
            - base_url: 优先 AI_BASE_URL，其次 API_URL
            - api_key: 优先 AI_API_KEY，其次 API_KEY
        - gemini:
            - model: 优先 GEMINI_MODEL，其次 AI_MODEL，再次 API_MODEL
            - base_url: 优先 GEMINI_BASE_URL，其次 AI_BASE_URL，其次 API_URL，最后回退 https://generativelanguage.googleapis.com
            - api_key: 优先 GEMINI_API_KEY，其次 AI_API_KEY，其次 API_KEY
        返回：
            {
              'provider': str,
              'model': str|None,
              'base_url': str|None,
              'api_key_present': bool,  # 不返回密钥，避免泄漏
              'ok': bool,
              'errors': list[str]
            }
        """
        provider = (os.getenv('AI_PROVIDER', '') or '').lower().strip() or 'newapi'

        # 通用变量
        ai_model = os.getenv('AI_MODEL') or None
        api_model = os.getenv('API_MODEL') or None
        ai_base_url = os.getenv('AI_BASE_URL') or None
        api_url = os.getenv('API_URL') or None
        ai_api_key = os.getenv('AI_API_KEY') or None
        api_key = os.getenv('API_KEY') or None

        model = None
        base_url = None
        key = None

        if provider in {"newapi", "openai", "deepseek"}:
            model = ai_model or api_model
            base_url = ai_base_url or api_url
            key = ai_api_key or api_key
        elif provider == "gemini":
            # 模型优先 GEMINI_MODEL -> AI_MODEL -> API_MODEL
            model = os.getenv('GEMINI_MODEL') or ai_model or api_model
            # API Key 优先 GEMINI_API_KEY -> AI_API_KEY -> API_KEY
            key = os.getenv('GEMINI_API_KEY') or ai_api_key or api_key
            # Base URL 优先 GEMINI_BASE_URL -> AI_BASE_URL -> API_URL -> 官方默认
            base_url = os.getenv('GEMINI_BASE_URL') or ai_base_url or api_url or "https://generativelanguage.googleapis.com"
        else:
            # 未知 provider，按 newapi 兼容处理
            model = ai_model or api_model
            base_url = ai_base_url or api_url
            key = ai_api_key or api_key

        errors = []
        if not model:
            errors.append("model_missing")
        if not base_url:
            errors.append("base_url_missing")
        if not key:
            errors.append("api_key_missing")

        return {
            'provider': provider,
            'model': model,
            'base_url': base_url,
            'api_key_present': bool(key),
            'ok': len(errors) == 0,
            'errors': errors,
        }

    @staticmethod
    def require_ai_config():
        """
        校验当前环境下的 AI 配置是否完整，缺失则抛出 ValueError。
        """
        conf = APIUtils.resolve_ai_config()
        if not conf.get('ok'):
            # 仅汇总必要信息，不包含密钥
            raise ValueError(f"AI 配置不完整: provider={conf.get('provider')}, model={conf.get('model')}, base_url={'set' if conf.get('base_url') else 'missing'}, api_key_present={conf.get('api_key_present')}, errors={conf.get('errors')}")
        return conf
