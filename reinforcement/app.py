from aiohttp import web
import socketio
import asyncio
from pong import Pong

GAMES = {}

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

async def index(request):
    """Serve the client-side application."""
    with open('static/index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.on('connect', namespace='/pong')
def connect(sid, environ):
    GAMES[sid] = Pong()
    print("connect ", sid)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_game_states)

async def send_game_states(sid):
    while True:
        sio.emit('game state', GAMES[sid].state(), room=sid)
        asyncio.sleep(0.01)


@sio.on('make move', namespace='/pong')
async def make_move(sid, data):
    GAMES[sid].play_action(data)

@sio.on('disconnect', namespace='/chat')
def disconnect(sid):
    del GAMES[sid]
    print('disconnect ', sid)

app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)
