import asyncio
import os
from utils.api_utils import APIUtils


async def _call_health():
    # 延迟导入以避免启动期环境干扰
    from web_server import health_ai
    return await health_ai()


def test_provider_newapi_with_ai_model(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'newapi')
    monkeypatch.setenv('AI_MODEL', 'gpt-4o-mini')
    monkeypatch.setenv('AI_BASE_URL', 'https://api.fake.com')
    monkeypatch.setenv('AI_API_KEY', 'sk-xxx')

    conf = APIUtils.resolve_ai_config()
    assert conf['provider'] == 'newapi'
    assert conf['model'] == 'gpt-4o-mini'
    assert conf['base_url'] == 'https://api.fake.com'
    assert conf['api_key_present'] is True
    assert conf['ok'] is True


def test_provider_gemini_with_gemini_model(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'gemini')
    monkeypatch.setenv('GEMINI_MODEL', 'gemini-1.5-flash')
    monkeypatch.setenv('GEMINI_API_KEY', 'gk-xxx')
    monkeypatch.setenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com')

    conf = APIUtils.resolve_ai_config()
    assert conf['provider'] == 'gemini'
    assert conf['model'] == 'gemini-1.5-flash'
    assert conf['base_url'].startswith('https://generativelanguage.googleapis.com')
    assert conf['api_key_present'] is True
    assert conf['ok'] is True


def test_provider_gemini_fallback_from_ai_model(monkeypatch):
    monkeypatch.setenv('AI_PROVIDER', 'gemini')
    # 不设置 GEMINI_MODEL，使用通用 AI_MODEL 回退
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    monkeypatch.setenv('AI_MODEL', 'gemini-1.5-flash')
    monkeypatch.setenv('GEMINI_API_KEY', 'gk-yyy')
    # 未设置基址时应回退到官方默认
    monkeypatch.delenv('GEMINI_BASE_URL', raising=False)
    monkeypatch.delenv('AI_BASE_URL', raising=False)
    monkeypatch.delenv('API_URL', raising=False)

    conf = APIUtils.resolve_ai_config()
    assert conf['provider'] == 'gemini'
    assert conf['model'] == 'gemini-1.5-flash'
    assert conf['base_url'].startswith('https://generativelanguage.googleapis.com')
    assert conf['ok'] is True


def test_missing_model_raises_and_health_reports(monkeypatch):
    # 设置 provider 与凭据，但完全缺失模型变量
    monkeypatch.setenv('AI_PROVIDER', 'newapi')
    monkeypatch.setenv('AI_BASE_URL', 'https://api.fake.com')
    monkeypatch.setenv('AI_API_KEY', 'sk-xxx')
    # 清除所有模型相关变量
    monkeypatch.delenv('AI_MODEL', raising=False)
    monkeypatch.delenv('API_MODEL', raising=False)
    monkeypatch.delenv('GEMINI_MODEL', raising=False)

    conf = APIUtils.resolve_ai_config()
    assert conf['ok'] is False
    assert 'model_missing' in conf['errors']

    # require_ai_config 应抛出配置异常
    try:
        APIUtils.require_ai_config()
        assert False, 'expect require_ai_config to raise'
    except ValueError as e:
        assert 'AI 配置不完整' in str(e)

    # 健康检查应返回错误状态与原因
    resp = asyncio.run(_call_health())
    assert resp['status'] == 'error'
    assert 'model_missing' in (resp.get('errors') or [])
