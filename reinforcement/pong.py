import numpy as np
import random

PLAYER1 = 1
PLAYER2 = 2
PADDLE_SIZE = 0.1
BALL_SPEED = 0.01
EPSILON = 0.02
NUDGE = 0.0025
UP_ACTION = 0
DOWN_ACTION = 1

class Pong():
    def __init__(self):
        self.paddle1_pos = 0.5
        self.paddle2_pos = 0.5
        self.score = 0
        self.reset_ball()

    def state(self):
        return (
            self.paddle1_pos,
            self.paddle2_pos,
            self.ball_pos[0],
            self.ball_pos[1],
            self.ball_vel[0],
            self.ball_vel[1]
        )

    def actions(self):
        return [UP_ACTION, DOWN_ACTION]

    def play_action(self, action):
        if action == UP_ACTION:
            self.nudge_up(PLAYER1)
        elif action == DOWN_ACTION:
            self.nudge_down(PLAYER2)

        self.computer_move()

        self.step()

    def computer_move(self):
        if self.ball_pos[0] < self.paddle2_pos:
            self.nudge_up(PLAYER2)
        elif self.ball_pos[0] > self.paddle2_pos:
            self.nudge_down(PLAYER2)

    def nudge_up(self, playernum):
        if playernum == 1:
            self.paddle1_pos = max(0, self.paddle1_pos - NUDGE)

        elif playernum == 2:
            self.paddle2_pos = max(0, self.paddle2_pos - NUDGE)

    def nudge_down(self, playernum):
        if playernum == PLAYER1:
            self.paddle1_pos = min(1, self.paddle1_pos + NUDGE)
        elif playernum == PLAYER2:
            self.paddle2_pos = min(1, self.paddle2_pos + NUDGE)

    def step(self):
        self.ball_pos += self.ball_vel
        self.check_if_point_scored()
        self.check_collision(PLAYER1)
        self.check_paddle_collision(PLAYER2)
        self.check_ball_collision()

    def check_if_point_scored(self):
        if self.ball_pos[1] < 0:
            self.score -= 1
            self.reset()
        elif self.ball_pos[1] > 1:
            self.score += 1
            self.reset()

    def check_paddle_collision(self, playernum):
        top, bottom = self.paddle_endpoints(playernum)

        if playernum == PLAYER1 and \
            self.ball_pos[1] < EPSILON and \
            top < self.ball_pos[0] and \
            bottom > self.ball_pos[0]:
            self.bounce()
        elif playernum == PLAYER2 and \
            1 - EPSILON < self.ball_pos[1] and \
            top < self.ball_pos[0] and \
            bottom > self.ball_pos[0]:
            self.bounce()

    def check_ball_collision(self):
        if self.ball_pos[0] < EPSILON or 1 - EPSILON < self.ball_pos[0]:
            self.ball_vel[0] *= -1

    def bounce(self):
        self.ball_vel[0] += random.random(-0.1, 0.1)
        self.ball_vel[1] *= -1
        self.ball_vel /= np.linalg.norm(self.ball_vel)
        self.ball_vel *= BALL_SPEED

    def paddle_endpoints(self, playernum):
        if playernum == PLAYER1:
            return (
                self.paddle1_pos - PADDLE_SIZE/2,
                self.paddle1_pos + PADDLE_SIZE/2)
        elif playernum == PLAYER2:
            return (
                self.paddle2_pos - PADDLE_SIZE/2,
                self.paddle2_pos + PADDLE_SIZE/2)

    def reset_ball(self):
        self.ball_pos = np.array([0.5, 0.5])
        self.ball_vel = np.array([0, BALL_SPEED])
