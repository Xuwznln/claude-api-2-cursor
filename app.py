import json
import logging

import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from config import Config
from openai_adapter import (
    anthropic_to_openai_response,
    anthropic_to_openai_stream_chunk,
    init_stream_state,
    cleanup_stream_state,
    openai_to_anthropic_request,
)

logger = logging.getLogger(__name__)

# Cursor 模型名 → Anthropic 模型名 映射
# 使用实际存在的模型名
MODEL_MAP = {
    'claude-4.6-sonnet-medium-thinking': 'claude-sonnet-4-5',
    'claude-4.6-sonnet-medium':          'claude-sonnet-4-5',
    'claude-4.6-opus-high-thinking':     'claude-opus-4-5',
    'claude-4.6-opus-high':              'claude-opus-4-5',
    'claude-4.6-haiku-low-thinking':     'claude-haiku-4-5',
    'claude-4.6-haiku-low':              'claude-haiku-4-5',
}

# thinking 推理强度 → budget_tokens 映射
# medium = 8000 tokens（Claude Code 官方推荐中等强度）
THINKING_BUDGET = {
    'low':    1024,
    'medium': 8000,
    'high':   16000,
}


def _get_global_thinking_config():
    """
    读取全局 thinking 配置（来自环境变量）。
    ENABLE_THINKING=true 时，对所有请求强制启用 thinking。
    THINKING_BUDGET 控制推理强度：low / medium / high。
    """
    if not Config.ENABLE_THINKING:
        return None
    budget = THINKING_BUDGET.get(Config.THINKING_BUDGET, THINKING_BUDGET['medium'])
    return {'type': 'enabled', 'budget_tokens': budget}


def _extract_thinking_config(original_model: str):
    """
    从 Cursor 模型名提取 thinking 配置。
    例如：claude-4.6-sonnet-medium-thinking → {type: enabled, budget_tokens: 8000}
    优先级：模型名中的强度 > 全局 ENABLE_THINKING 环境变量。
    """
    lower = original_model.lower()
    # 模型名含 thinking 后缀 → 按强度关键词匹配
    if 'thinking' in lower:
        for intensity, budget in THINKING_BUDGET.items():
            if f'-{intensity}-thinking' in lower:
                return {'type': 'enabled', 'budget_tokens': budget}
        # 有 thinking 后缀但未指定强度 → 默认 medium
        return {'type': 'enabled', 'budget_tokens': THINKING_BUDGET['medium']}
    # 模型名不含 thinking → 检查全局开关
    return _get_global_thinking_config()


def _log_request_exception(tag, e):
    """请求失败时打印完整错误：异常信息、堆栈、以及上游响应体（若有）"""
    logger.error('%s request error: %s', tag, e, exc_info=True)
    if getattr(e, 'response', None) is not None:
        try:
            body = e.response.content.decode('utf-8', errors='replace')
            logger.error('%s upstream response %s: %s', tag, e.response.status_code, body)
        except Exception:
            logger.error('%s upstream response %s: (body decode failed)', tag, e.response.status_code)


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.before_request
    def check_access_key():
        """接入鉴权：校验 ACCESS_API_KEY"""
        if not Config.ACCESS_API_KEY:
            return  # 未配置则不鉴权
        if request.path == '/health':
            return  # 健康检查跳过鉴权

        auth = request.headers.get('Authorization', '')
        token = ''
        if auth.startswith('Bearer '):
            token = auth[7:]
        if not token:
            token = request.headers.get('x-api-key', '')

        if token != Config.ACCESS_API_KEY:
            logger.warning(f'[auth] rejected {request.path}')
            return jsonify({
                'error': {'message': 'Invalid API key', 'type': 'authentication_error'}
            }), 401

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok', 'target': Config.PROXY_TARGET_URL})

    @app.route('/v1/models', methods=['GET'])
    def list_models():
        """Cursor 启动时会调用此接口获取可用模型列表"""
        models = [
            {'id': 'claude-sonnet-4-5-20250929', 'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-sonnet-4-5',          'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-sonnet-4-20250514',   'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-opus-4-5-20251101',   'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-opus-4-5',            'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-haiku-4-5-20251001',  'object': 'model', 'owned_by': 'anthropic'},
            {'id': 'claude-haiku-4-5',           'object': 'model', 'owned_by': 'anthropic'},
        ]
        return jsonify({'object': 'list', 'data': models})

    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        """OpenAI 兼容接口 — 主路由"""
        payload = request.get_json(force=True)
        is_stream = payload.get('stream', False)
        model = payload.get('model', 'unknown')
        msg_count = len(payload.get('messages', []))
        logger.info(f'[chat] model={model} stream={is_stream} messages={msg_count}')

        # 记录每条消息的摘要
        for i, msg in enumerate(payload.get('messages', [])):
            role = msg.get('role', '?')
            content = msg.get('content')
            content_type = type(content).__name__
            has_tc = 'tool_calls' in msg
            tc_count = len(msg.get('tool_calls', []))
            tc_id = msg.get('tool_call_id', '')
            if isinstance(content, list):
                types = [p.get('type','?') if isinstance(p,dict) else 'str' for p in content]
                content_info = f'list[{len(content)}] types={types}'
            elif isinstance(content, str):
                content_info = f'str[{len(content)}]'
            elif content is None:
                content_info = 'None'
            else:
                content_info = content_type
            extra = ''
            if has_tc:
                extra += f' tool_calls={tc_count}'
            if tc_id:
                extra += f' tool_call_id={tc_id}'
            logger.info(f'[chat]   msg[{i}] role={role} content={content_info}{extra}')

        # 模型名映射 + thinking 配置提取
        original_model = payload.get('model', '')
        thinking_config = _extract_thinking_config(original_model)
        if original_model in MODEL_MAP:
            payload = {**payload, 'model': MODEL_MAP[original_model]}
            logger.info(f'[chat] model mapped: {original_model} -> {payload["model"]}')

        # 转换请求
        anthropic_payload = openai_to_anthropic_request(payload)

        # 注入 thinking 参数（推理强度 medium = 8000 tokens）
        if thinking_config:
            anthropic_payload['thinking'] = thinking_config
            # thinking 要求 max_tokens 至少大于 budget_tokens
            budget = thinking_config['budget_tokens']
            current_max = anthropic_payload.get('max_tokens', 8096)
            if current_max <= budget:
                anthropic_payload['max_tokens'] = budget + 4096
            logger.info(f'[chat] thinking enabled: budget={budget} max_tokens={anthropic_payload["max_tokens"]}')

        logger.debug(f'[chat] anthropic_payload: {json.dumps(anthropic_payload, ensure_ascii=False)}')

        # 准备请求头
        headers = _prepare_headers(thinking_enabled=thinking_config is not None)
        headers['Content-Type'] = 'application/json'

        target_url = f'{Config.PROXY_TARGET_URL.rstrip("/")}/v1/messages'

        if is_stream:
            anthropic_payload['stream'] = True
            return _handle_stream(target_url, headers, anthropic_payload)
        else:
            anthropic_payload['stream'] = False
            return _handle_non_stream(target_url, headers, anthropic_payload)

    @app.route('/v1/messages', methods=['POST'])
    def messages_passthrough():
        """Anthropic 原生格式透传（含模型映射和 thinking 注入）"""
        payload = request.get_json(force=True)
        original_model = payload.get('model', 'unknown')
        is_stream = payload.get('stream', False)
        logger.info(f'[passthrough] model={original_model} stream={is_stream}')

        # 模型名映射
        thinking_config = _extract_thinking_config(original_model)
        if original_model in MODEL_MAP:
            payload = {**payload, 'model': MODEL_MAP[original_model]}
            logger.info(f'[passthrough] model mapped: {original_model} -> {payload["model"]}')

        # 注入 thinking 参数
        if thinking_config and 'thinking' not in payload:
            payload = {**payload, 'thinking': thinking_config}
            budget = thinking_config['budget_tokens']
            current_max = payload.get('max_tokens', 8096)
            if current_max <= budget:
                payload = {**payload, 'max_tokens': budget + 4096}
            logger.info(f'[passthrough] thinking enabled: budget={budget}')

        headers = _prepare_headers(thinking_enabled=thinking_config is not None)
        headers['Content-Type'] = 'application/json'

        target_url = f'{Config.PROXY_TARGET_URL.rstrip("/")}/v1/messages'
        is_stream = payload.get('stream', False)

        try:
            resp = requests.post(
                target_url,
                headers=headers,
                json=payload,
                timeout=Config.API_TIMEOUT,
                stream=is_stream,
            )

            if is_stream:
                def generate():
                    for line in resp.iter_lines():
                        if line:
                            yield line.decode('utf-8', errors='replace') + '\n\n'

                return Response(generate(), content_type='text/event-stream')
            else:
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get('Content-Type', 'application/json'),
                )
        except requests.RequestException as e:
            _log_request_exception('[passthrough]', e)
            return jsonify({'error': {'message': str(e), 'type': 'proxy_error'}}), 502

    def _handle_non_stream(target_url, headers, anthropic_payload):
        """处理非流式请求"""
        CONNECT_TIMEOUT = 10
        try:
            resp = requests.post(
                target_url,
                headers=headers,
                json=anthropic_payload,
                timeout=(CONNECT_TIMEOUT, Config.API_TIMEOUT),  # (connect, read)
            )

            if resp.status_code != 200:
                error_body = resp.content.decode('utf-8', errors='replace')
                logger.warning(f'[chat] upstream error {resp.status_code}: %s', error_body)
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get('Content-Type', 'application/json'),
                )

            anthropic_data = resp.json()
            openai_response = anthropic_to_openai_response(anthropic_data)
            usage = openai_response.get('usage', {})
            logger.info(f'[chat] done prompt={usage.get("prompt_tokens", 0)} completion={usage.get("completion_tokens", 0)}')
            return jsonify(openai_response)

        except requests.exceptions.ConnectTimeout:
            logger.error(f'[chat] connect timeout to {target_url}')
            return jsonify({'error': {'message': 'Connection timeout: check PROXY_TARGET_URL.', 'type': 'connect_timeout'}}), 502
        except requests.exceptions.ReadTimeout:
            logger.error(f'[chat] read timeout after {Config.API_TIMEOUT}s')
            return jsonify({'error': {'message': f'Read timeout after {Config.API_TIMEOUT}s.', 'type': 'read_timeout'}}), 502
        except requests.RequestException as e:
            _log_request_exception('[chat]', e)
            return jsonify({'error': {'message': str(e), 'type': 'proxy_error'}}), 502

    def _handle_stream(target_url, headers, anthropic_payload):
        """处理流式请求"""
        request_id = f'chatcmpl-stream-{id(request)}'
        # 连接超时 10s，读取超时使用 API_TIMEOUT（默认 300s）
        # 分离两个值避免连接挂起导致整个请求长时间阻塞
        CONNECT_TIMEOUT = 10

        def generate():
            init_stream_state(request_id)
            event_type = 'message_start'  # 初始化防止 NameError
            try:
                resp = requests.post(
                    target_url,
                    headers=headers,
                    json=anthropic_payload,
                    timeout=(CONNECT_TIMEOUT, Config.API_TIMEOUT),  # (connect, read)
                    stream=True,
                )

                if resp.status_code != 200:
                    error_body = resp.content.decode('utf-8', errors='replace')
                    logger.warning(f'[stream] upstream error {resp.status_code}: %s', error_body)
                    error_chunk = json.dumps({
                        'error': {
                            'message': f'Upstream error {resp.status_code}: {error_body}',
                            'type': 'upstream_error',
                        }
                    })
                    yield f'data: {error_chunk}\n\n'
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode('utf-8', errors='replace')

                    if decoded.startswith('event:'):
                        event_type = decoded[6:].strip()
                        continue

                    if decoded.startswith('data:'):
                        data_str = decoded[5:].strip()
                        if not data_str:
                            continue
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        logger.debug(f'[stream] event={event_type} data_keys={list(event_data.keys()) if isinstance(event_data, dict) else "?"}')
                        if event_type == 'content_block_start':
                            block = event_data.get('content_block', {})
                            logger.info(f'[stream] content_block_start type={block.get("type")} name={block.get("name", "")}')

                        chunks = anthropic_to_openai_stream_chunk(
                            event_type, event_data, request_id
                        )
                        for chunk_str in chunks:
                            yield f'data: {chunk_str}\n\n'

                yield 'data: [DONE]\n\n'

            except requests.exceptions.ConnectTimeout:
                # 连接阶段超时（10s内无法建立连接）
                logger.error(f'[stream] connect timeout to {target_url} after {CONNECT_TIMEOUT}s')
                error_chunk = json.dumps({
                    'error': {
                        'message': f'Connection timeout: could not connect to upstream within {CONNECT_TIMEOUT}s. Check PROXY_TARGET_URL.',
                        'type': 'connect_timeout',
                    }
                })
                yield f'data: {error_chunk}\n\n'
            except requests.exceptions.ReadTimeout:
                # 读取阶段超时（已连接但响应超时）
                logger.error(f'[stream] read timeout after {Config.API_TIMEOUT}s')
                error_chunk = json.dumps({
                    'error': {
                        'message': f'Read timeout after {Config.API_TIMEOUT}s. Try increasing API_TIMEOUT.',
                        'type': 'read_timeout',
                    }
                })
                yield f'data: {error_chunk}\n\n'
            except requests.RequestException as e:
                _log_request_exception('[stream]', e)
                error_chunk = json.dumps({
                    'error': {'message': str(e), 'type': 'proxy_error'}
                })
                yield f'data: {error_chunk}\n\n'
            finally:
                cleanup_stream_state(request_id)

        return Response(
            generate(),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            },
        )

    return app


def _prepare_headers(thinking_enabled: bool = False):
    """准备请求头，注入 API Key 及可选的 thinking beta header"""
    headers = {
        'anthropic-version': '2023-06-01',
    }
    # thinking 需要开启 beta 特性
    if thinking_enabled:
        headers['anthropic-beta'] = 'interleaved-thinking-2025-05-14'
    key = Config.PROXY_API_KEY
    if key.startswith('sk-'):
        headers['x-api-key'] = key
    else:
        headers['Authorization'] = f'Bearer {key}'
    return headers
