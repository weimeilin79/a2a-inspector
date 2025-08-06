import logging
from typing import Any
from uuid import uuid4

import httpx
import socketio
import validators

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    TextPart,
)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ==============================================================================
# Setup
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount('/socket.io', socket_app)

app.mount('/static', StaticFiles(directory='../frontend/public'), name='static')
templates = Jinja2Templates(directory='../frontend/public')

# ==============================================================================
# State Management
# ==============================================================================

clients: dict[str, tuple[httpx.AsyncClient, A2AClient, AgentCard]] = {}

# ==============================================================================
# Socket.IO Event Helpers
# ==============================================================================


async def _emit_debug_log(
    sid: str, event_id: str, log_type: str, data: Any
) -> None:
    """Helper to emit a structured debug log event to the client."""
    await sio.emit(
        'debug_log', {'type': log_type, 'data': data, 'id': event_id}, to=sid
    )


async def _process_a2a_response(
    result: SendMessageResponse | SendStreamingMessageResponse,
    sid: str,
    request_id: str,
) -> None:
    """Processes a response from the A2A client, validates it, and emits events.

    Handles both success and error responses.
    """
    if isinstance(result.root, JSONRPCErrorResponse):
        error_data = result.root.error.model_dump(exclude_none=True)
        await _emit_debug_log(sid, request_id, 'error', error_data)
        await sio.emit(
            'agent_response',
            {'error': error_data.get('message', 'Unknown error'), 'id': request_id},
            to=sid,
        )
        return

    # Success case
    event = result.root.result
    response_id = getattr(event, 'id', request_id)
    response_data = event.model_dump(exclude_none=True)
    response_data['id'] = response_id

    # START of updated logic
    final_message_to_display = None
    status_info = response_data.get('status', {})

    # Priority 1: Check for a 'failed' status message.
    if status_info.get('state') == 'failed':
        try:
            final_message_to_display = status_info['message']['parts'][0]['text']
        except (KeyError, IndexError):
            final_message_to_display = "Task failed with an unknown error."

    # Priority 2: Check for a final message in the top-level 'artifacts' array.
    # This is often the definitive result of a task.
    if not final_message_to_display:
        try:
            artifacts = response_data.get('artifacts', [])
            if artifacts and artifacts[0].get('parts'):
                text_part = artifacts[0]['parts'][0].get('text')
                if text_part:
                    final_message_to_display = text_part
        except (KeyError, IndexError):
            pass # Ignore if structure is not as expected

    # Priority 3: If no artifact, search history for the most "important" message.
    # Heuristic: The last agent message containing bold text (**) is often an action summary.
    if not final_message_to_display and response_data.get('kind') == 'task' and 'history' in response_data:
        for message in reversed(response_data.get('history', [])):
            if message.get('role') == 'agent':
                parts = message.get('parts', [])
                if parts and parts[0].get('kind') == 'text':
                    text = parts[0].get('text', '')
                    if '**' in text:
                        final_message_to_display = text
                        break  # Found the most recent "important" message

    # Priority 4: If no "important" message was found, fall back to the very last agent text message.
    if not final_message_to_display and response_data.get('kind') == 'task' and 'history' in response_data:
        for message in reversed(response_data.get('history', [])):
            if message.get('role') == 'agent':
                parts = message.get('parts', [])
                if parts and parts[0].get('kind') == 'text' and parts[0].get('text'):
                    final_message_to_display = parts[0]['text']
                    break

    if final_message_to_display:
        response_data['final_message'] = final_message_to_display
    # END of updated logic

    validation_errors = validators.validate_message(response_data)
    response_data['validation_errors'] = validation_errors

    await _emit_debug_log(sid, response_id, 'response', response_data)
    await sio.emit('agent_response', response_data, to=sid)


# ==============================================================================
# FastAPI Routes & Other Handlers (NO CHANGES BELOW THIS LINE)
# ==============================================================================


@app.get('/', response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the main index.html page."""
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/agent-card')
async def get_agent_card(request: Request) -> JSONResponse:
    """Fetch and validate the agent card from a given URL."""
    try:
        request_data = await request.json()
        agent_url = request_data.get('url')
        sid = request_data.get('sid')
        if not agent_url or not sid:
            return JSONResponse(
                content={'error': 'Agent URL and SID are required.'},
                status_code=400,
            )
    except Exception:
        logger.warning('Failed to parse JSON from /agent-card request.')
        return JSONResponse(
            content={'error': 'Invalid request body.'}, status_code=400
        )
    await _emit_debug_log(
        sid,
        'http-agent-card',
        'request',
        {'endpoint': '/agent-card', 'payload': request_data},
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            card_resolver = A2ACardResolver(client, agent_url)
            card = await card_resolver.get_agent_card()
        card_data = card.model_dump(exclude_none=True)
        validation_errors = validators.validate_agent_card(card_data)
        response_data = {
            'card': card_data,
            'validation_errors': validation_errors,
        }
        response_status = 200
    except httpx.RequestError as e:
        logger.error(
            f'Failed to connect to agent at {agent_url}', exc_info=True
        )
        response_data = {'error': f'Failed to connect to agent: {e}'}
        response_status = 502
    except Exception as e:
        logger.error('An internal server error occurred', exc_info=True)
        response_data = {'error': f'An internal server error occurred: {e}'}
        response_status = 500
    await _emit_debug_log(
        sid,
        'http-agent-card',
        'response',
        {'status': response_status, 'payload': response_data},
    )
    return JSONResponse(content=response_data, status_code=response_status)


@sio.on('connect')
async def handle_connect(sid: str, environ: dict[str, Any]) -> None:
    logger.info(f'Client connected: {sid}, environment: {environ}')


@sio.on('disconnect')
async def handle_disconnect(sid: str) -> None:
    logger.info(f'Client disconnected: {sid}')
    if sid in clients:
        httpx_client, _, _ = clients.pop(sid)
        await httpx_client.aclose()
        logger.info(f'Cleaned up client for {sid}')


@sio.on('initialize_client')
async def handle_initialize_client(sid: str, data: dict[str, Any]) -> None:
    agent_url = data.get('url')
    if not agent_url:
        await sio.emit(
            'client_initialized',
            {'status': 'error', 'message': 'Agent URL is required.'},
            to=sid,
        )
        return
    try:
        httpx_client = httpx.AsyncClient(timeout=600.0)
        card_resolver = A2ACardResolver(httpx_client, str(agent_url))
        card = await card_resolver.get_agent_card()
        a2a_client = A2AClient(httpx_client, agent_card=card)
        clients[sid] = (httpx_client, a2a_client, card)
        await sio.emit('client_initialized', {'status': 'success'}, to=sid)
    except Exception as e:
        logger.error(
            f'Failed to initialize client for {sid}: {e}', exc_info=True
        )
        await sio.emit(
            'client_initialized', {'status': 'error', 'message': str(e)}, to=sid
        )


@sio.on('send_message')
async def handle_send_message(sid: str, json_data: dict[str, Any]) -> None:
    message_text = json_data.get('message')
    message_id = json_data.get('id', str(uuid4()))
    if sid not in clients:
        await sio.emit(
            'agent_response',
            {'error': 'Client not initialized.', 'id': message_id},
            to=sid,
        )
        return
    _, a2a_client, card = clients[sid]
    message = Message(
        role=Role.user,
        parts=[TextPart(text=str(message_text))],
        messageId=str(uuid4()),
    )
    payload = MessageSendParams(
        message=message,
        configuration=MessageSendConfiguration(
            acceptedOutputModes=['text/plain', 'video/mp4']
        ),
    )
    supports_streaming = (
        hasattr(card.capabilities, 'streaming')
        and card.capabilities.streaming is True
    )
    try:
        if supports_streaming:
            stream_request = SendStreamingMessageRequest(
                id=message_id,
                method='message/stream',
                jsonrpc='2.0',
                params=payload,
            )
            await _emit_debug_log(
                sid,
                message_id,
                'request',
                stream_request.model_dump(exclude_none=True),
            )
            response_stream = a2a_client.send_message_streaming(stream_request)
            async for stream_result in response_stream:
                await _process_a2a_response(stream_result, sid, message_id)
        else:
            send_message_request = SendMessageRequest(
                id=message_id,
                method='message/send',
                jsonrpc='2.0',
                params=payload,
            )
            await _emit_debug_log(
                sid,
                message_id,
                'request',
                send_message_request.model_dump(exclude_none=True),
            )
            send_result = await a2a_client.send_message(send_message_request)
            await _process_a2a_response(send_result, sid, message_id)
    except Exception as e:
        logger.error(f'Failed to send message for sid {sid}', exc_info=True)
        await sio.emit(
            'agent_response',
            {'error': f'Failed to send message: {e}', 'id': message_id},
            to=sid,
        )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app:app', host='127.0.0.1', port=5001, reload=True)
