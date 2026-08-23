import os
import importlib
import pytest

mod = importlib.import_module('ai_providers')
call_best = mod.call_best_provider


def test_stub_when_no_providers(monkeypatch):
    # Ensure no provider envs are set
    monkeypatch.delenv('ENABLE_OPENAI', raising=False)
    monkeypatch.delenv('ENABLE_GROQ', raising=False)
    monkeypatch.delenv('ENABLE_OPENROUTER', raising=False)
    monkeypatch.delenv('AI_USE_MOCK', raising=False)
    res = call_best({})
    assert isinstance(res, dict)
    assert res.get('provider') == 'stub'
    assert res.get('ok') is True


def test_mock_per_call(monkeypatch):
    monkeypatch.delenv('AI_USE_MOCK', raising=False)
    monkeypatch.delenv('ENABLE_OPENAI', raising=False)
    payload = {'mock': True, 'prompt': 'hello test'}
    res = call_best(payload)
    assert res.get('provider') in ('mock', 'mock')
    assert res.get('ok') is True
    assert 'mocked-response' in res.get('text', '')


def test_mock_error_raises(monkeypatch):
    monkeypatch.delenv('AI_USE_MOCK', raising=False)
    monkeypatch.delenv('ENABLE_OPENAI', raising=False)
    with pytest.raises(RuntimeError):
        call_best({'mock_error': True})


def test_forced_prompt_error(monkeypatch):
    monkeypatch.delenv('AI_USE_MOCK', raising=False)
    monkeypatch.delenv('ENABLE_OPENAI', raising=False)
    with pytest.raises(RuntimeError):
        call_best({'prompt': 'will error'})


def test_provider_order_respected(monkeypatch):
    # Enable both providers but set order to prefer GROQ
    monkeypatch.delenv('AI_USE_MOCK', raising=False)
    monkeypatch.setenv('ENABLE_OPENAI', 'true')
    monkeypatch.setenv('ENABLE_GROQ', 'true')
    monkeypatch.setenv('AI_PROVIDER_ORDER', 'GROQ,OPENAI')
    res = call_best({'prompt': 'ok'})
    # our placeholder groq implementation returns provider 'groq'
    assert res.get('provider') == 'groq'
