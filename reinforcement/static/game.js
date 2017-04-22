let socket = io.connect('http://localhost:8080/pong');

socket.on('game state', (gameState) => {
  console.log(gameState);
});
