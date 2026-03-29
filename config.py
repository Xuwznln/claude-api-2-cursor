import os


class Config:
    PROXY_TARGET_URL = os.getenv('PROXY_TARGET_URL', 'https://api.anthropic.com')
    PROXY_API_KEY = os.getenv('PROXY_API_KEY', '')
    PROXY_PORT = int(os.getenv('PROXY_PORT', '3029'))
    API_TIMEOUT = int(os.getenv('API_TIMEOUT', '300'))
    ACCESS_API_KEY = os.getenv('ACCESS_API_KEY', '')

    # Thinking 配置
    # ENABLE_THINKING=true  → 所有请求强制开启 thinking（推荐用于 opus 模型）
    # THINKING_BUDGET=medium → 推理强度：low(1024) / medium(8000) / high(16000)
    ENABLE_THINKING: bool = os.getenv('ENABLE_THINKING', 'false').lower() in ('true', '1', 'yes')
    THINKING_BUDGET: str = os.getenv('THINKING_BUDGET', 'medium').lower()
