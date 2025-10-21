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
