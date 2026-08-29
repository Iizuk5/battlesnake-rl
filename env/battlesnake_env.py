"""
Custom Gymnasium environment simulating Battlesnake rules, for RL training.

This does NOT depend on any external Battlesnake library — the rules are
simple enough (and stable/well-documented) that we implement them directly.
This means training runs entirely locally with no network calls, no server,
and no fragile third-party dependency in the loop.

Rules implemented (standard Battlesnake ruleset, simplified to 1v1):
  - Grid board (default 11x11, matches the standard competitive size)
  - Each snake starts at health 100, loses 1 health per turn
  - Eating food resets health to 100 and grows the snake by 1
  - A snake dies if it: runs out of health, hits a wall, hits its own body,
    hits the opponent's body, or (on a head-to-head collision) is not the
    longer snake
  - New food spawns periodically at a random empty cell

This is "single-agent" from the trainee's point of view: our snake is
controlled by the RL policy, the opponent by a simple heuristic (see
opponent_policy below) — swap in a stronger opponent, or self-play, once
the basic pipeline is working.
"""

import random
from enum import IntEnum

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class Move(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_DELTAS = {
    Move.UP: (0, 1),
    Move.DOWN: (0, -1),
    Move.LEFT: (-1, 0),
    Move.RIGHT: (1, 0),
}


class Snake:
    def __init__(self, body, health=100):
        self.body = list(body)  # list of (x, y), body[0] is the head
        self.health = health
        self.alive = True

    @property
    def head(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)


class BattlesnakeEnv(gym.Env):
    """
    Single-agent Gymnasium env: our snake vs one heuristic opponent snake,
    on a standard-size board.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, board_size=11, max_turns=300, opponent_policy=None):
        super().__init__()
        self.board_size = board_size
        self.max_turns = max_turns
        self.opponent_policy = opponent_policy or self._default_opponent_policy

        # Observation: a fixed-size feature vector describing the local
        # situation around our snake's head, plus some global info.
        # See _get_obs() for exactly what's encoded — kept simple to start.
        self.observation_space = Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        self.action_space = Discrete(4)  # UP, DOWN, LEFT, RIGHT

        self.turn = 0
        self.food = set()
        self.us = None
        self.opponent = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn = 0
        mid = self.board_size // 2

        # Start the two snakes on opposite sides of the board.
        self.us = Snake(body=[(2, mid), (1, mid), (0, mid)])
        self.opponent = Snake(body=[(self.board_size - 3, mid), (self.board_size - 2, mid), (self.board_size - 1, mid)])

        self.food = set()
        self._spawn_food(count=3)

        return self._get_obs(), {}

    def step(self, action):
        self.turn += 1
        our_move = Move(action)
        opp_move = self.opponent_policy(self, self.opponent, self.us)

        self._apply_move(self.us, our_move)
        self._apply_move(self.opponent, opp_move)

        self._resolve_collisions()
        self._maybe_spawn_food()

        reward = self._calc_reward()
        terminated = not self.us.alive or not self.opponent.alive
        truncated = self.turn >= self.max_turns

        return self._get_obs(), reward, terminated, truncated, {}

    # ---- internals ----

    def _apply_move(self, snake: Snake, move: Move):
        if not snake.alive:
            return
        dx, dy = _DELTAS[move]
        new_head = (snake.head[0] + dx, snake.head[1] + dy)
        ate = new_head in self.food
        snake.body.insert(0, new_head)
        if ate:
            self.food.discard(new_head)
            snake.health = 100
        else:
            snake.body.pop()
            snake.health -= 1

    def _resolve_collisions(self):
        for snake, other in ((self.us, self.opponent), (self.opponent, self.us)):
            if not snake.alive:
                continue
            hx, hy = snake.head
            # Wall collision
            if not (0 <= hx < self.board_size and 0 <= hy < self.board_size):
                snake.alive = False
                continue
            # Starvation
            if snake.health <= 0:
                snake.alive = False
                continue
            # Self collision (head hit own body, excluding the head itself)
            if snake.head in snake.body[1:]:
                snake.alive = False
                continue
            # Collision with other snake's body (excluding other's head — handled below)
            if other.alive and snake.head in other.body[1:]:
                snake.alive = False
                continue

        # Head-to-head collision: shorter snake (or equal length) dies.
        if self.us.alive and self.opponent.alive and self.us.head == self.opponent.head:
            if self.us.length <= self.opponent.length:
                self.us.alive = False
            if self.opponent.length <= self.us.length:
                self.opponent.alive = False

    def _spawn_food(self, count=1):
        occupied = set(self.us.body) | set(self.opponent.body) | self.food
        empty = [
            (x, y)
            for x in range(self.board_size)
            for y in range(self.board_size)
            if (x, y) not in occupied
        ]
        random.shuffle(empty)
        for cell in empty[:count]:
            self.food.add(cell)

    def _maybe_spawn_food(self):
        # Roughly matches official Battlesnake's food spawn cadence: spawn
        # if below a minimum count, otherwise spawn with a small chance each turn.
        if len(self.food) < 1:
            self._spawn_food(count=1)
        elif random.random() < 0.15:
            self._spawn_food(count=1)

    def _calc_reward(self):
        reward = 0.0
        if not self.us.alive and not self.opponent.alive:
            reward = 0.0  # simultaneous death, draw
        elif not self.us.alive:
            reward = -10.0  # we died
        elif not self.opponent.alive:
            reward = 10.0  # opponent died, we win
        else:
            reward = 0.01  # small per-turn survival incentive
        return reward

    def _get_obs(self):
        """
        Encodes, from our snake's point of view:
          - Danger in each of the 4 directions (wall/body collision next turn): 4 values
          - Direction to nearest food (dx, dy, normalized): 2 values
          - Direction to opponent's head (dx, dy, normalized): 2 values
          - Our health / 100, opponent health / 100: 2 values
          - Our length, opponent length (normalized by board_size): 2 values
          - Which direction is directly toward the board center (dx, dy, normalized): 2 values
          - Padding for future features: remaining slots filled with 0

        Kept intentionally simple to get an end-to-end pipeline training
        fast. See README "Next steps" for how to expand this (e.g. a small
        local grid patch around the head instead of just danger flags).
        """
        obs = np.zeros(20, dtype=np.float32)

        hx, hy = self.us.head
        occupied = set(self.us.body) | set(self.opponent.body)

        for i, move in enumerate([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]):
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            danger = (
                not (0 <= nx < self.board_size and 0 <= ny < self.board_size)
                or (nx, ny) in occupied
            )
            obs[i] = 1.0 if danger else 0.0

        if self.food:
            fx, fy = min(self.food, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))
            obs[4] = np.clip((fx - hx) / self.board_size, -1, 1)
            obs[5] = np.clip((fy - hy) / self.board_size, -1, 1)

        ox, oy = self.opponent.head
        obs[6] = np.clip((ox - hx) / self.board_size, -1, 1)
        obs[7] = np.clip((oy - hy) / self.board_size, -1, 1)

        obs[8] = self.us.health / 100.0
        obs[9] = self.opponent.health / 100.0
        obs[10] = self.us.length / (self.board_size * self.board_size)
        obs[11] = self.opponent.length / (self.board_size * self.board_size)

        mid = self.board_size / 2
        obs[12] = np.clip((mid - hx) / self.board_size, -1, 1)
        obs[13] = np.clip((mid - hy) / self.board_size, -1, 1)

        return obs

    @staticmethod
    def _default_opponent_policy(env, opponent: Snake, other: Snake):
        """
        Simple heuristic opponent: pick a random move that doesn't
        immediately kill it (wall/self/other-body collision), if one
        exists; otherwise pick randomly (it's going to die anyway).
        """
        hx, hy = opponent.head
        occupied = set(opponent.body) | set(other.body)
        safe_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < env.board_size and 0 <= ny < env.board_size and (nx, ny) not in occupied:
                safe_moves.append(move)
        return random.choice(safe_moves) if safe_moves else random.choice(list(Move))

    def render(self):
        grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        for (x, y) in self.food:
            grid[y][x] = "F"
        for (x, y) in self.opponent.body[1:]:
            grid[y][x] = "o"
        if self.opponent.alive:
            ox, oy = self.opponent.head
            grid[oy][ox] = "O"
        for (x, y) in self.us.body[1:]:
            grid[y][x] = "s"
        if self.us.alive:
            ux, uy = self.us.head
            grid[uy][ux] = "S"
        return "\n".join("".join(row) for row in reversed(grid))
