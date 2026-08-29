"""
Custom Gymnasium environment simulating Battlesnake rules, for RL training.
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


def flood_fill_size(start, occupied, board_size, cap=None):
    """
    Breadth-first count of cells reachable from `start` without crossing
    a wall or an occupied cell. Standalone so both training and serving
    use the exact same logic.
    """
    if start in occupied or not (0 <= start[0] < board_size and 0 <= start[1] < board_size):
        return 0
    cap = cap or (board_size * board_size)
    seen = {start}
    queue = [start]
    count = 0
    head = 0
    while head < len(queue) and count < cap:
        x, y = queue[head]
        head += 1
        count += 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < board_size
                and 0 <= ny < board_size
                and (nx, ny) not in occupied
                and (nx, ny) not in seen
            ):
                seen.add((nx, ny))
                queue.append((nx, ny))
    return count


class Snake:
    def __init__(self, body, health=100):
        self.body = list(body)
        self.health = health
        self.alive = True

    @property
    def head(self):
        return self.body[0]

    @property
    def length(self):
        return len(self.body)


class BattlesnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, board_size=11, max_turns=300, opponent_policy=None, opponent_policies=None, opponent_weights=None):
        super().__init__()
        self.board_size = board_size
        self.max_turns = max_turns

        if opponent_policy is not None:
            self._fixed_opponent_policy = opponent_policy
            self._opponent_policy_pool = None
        else:
            self._fixed_opponent_policy = None
            self._opponent_policy_pool = opponent_policies or [
                self._random_safe_opponent_policy,
                self._food_seeking_opponent_policy,
                self._flood_fill_opponent_policy,
            ]
        self._opponent_weights = opponent_weights
        self.opponent_policy = self._fixed_opponent_policy or self._random_safe_opponent_policy

        self.observation_space = Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        self.action_space = Discrete(4)

        self.turn = 0
        self.food = set()
        self.us = None
        self.opponent = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.turn = 0
        mid = self.board_size // 2
        if self._fixed_opponent_policy is None:
            self.opponent_policy = random.choices(
                self._opponent_policy_pool, weights=self._opponent_weights, k=1
            )[0]

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

    def _apply_move(self, snake, move):
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
            if not (0 <= hx < self.board_size and 0 <= hy < self.board_size):
                snake.alive = False
                continue
            if snake.health <= 0:
                snake.alive = False
                continue
            if snake.head in snake.body[1:]:
                snake.alive = False
                continue
            if other.alive and snake.head in other.body[1:]:
                snake.alive = False
                continue

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
        if len(self.food) < 1:
            self._spawn_food(count=1)
        elif random.random() < 0.15:
            self._spawn_food(count=1)

    def _calc_reward(self):
        reward = 0.0
        if not self.us.alive and not self.opponent.alive:
            reward = 0.0
        elif not self.us.alive:
            reward = -10.0
        elif not self.opponent.alive:
            reward = 10.0
        else:
            reward = 0.01
        return reward

    def _flood_fill_size(self, start, occupied, cap=None):
        return flood_fill_size(start, occupied, self.board_size, cap=cap)

    def _get_obs(self):
        obs = np.zeros(24, dtype=np.float32)

        hx, hy = self.us.head
        our_occupied = set(self.us.body) | set(self.opponent.body)
        floodfill_occupied = (set(self.us.body[:-1]) | set(self.opponent.body[:-1])) - {(hx, hy)}

        for i, move in enumerate([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]):
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            danger = (
                not (0 <= nx < self.board_size and 0 <= ny < self.board_size)
                or (nx, ny) in our_occupied
            )
            obs[i] = 1.0 if danger else 0.0

            space = self._flood_fill_size((nx, ny), floodfill_occupied, cap=30)
            obs[4 + i] = min(space / 30.0, 1.0)

        food_sorted = sorted(self.food, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))
        if len(food_sorted) >= 1:
            fx, fy = food_sorted[0]
            obs[8] = np.clip((fx - hx) / self.board_size, -1, 1)
            obs[9] = np.clip((fy - hy) / self.board_size, -1, 1)
        if len(food_sorted) >= 2:
            fx2, fy2 = food_sorted[1]
            obs[10] = np.clip((fx2 - hx) / self.board_size, -1, 1)
            obs[11] = np.clip((fy2 - hy) / self.board_size, -1, 1)

        ox, oy = self.opponent.head
        obs[12] = np.clip((ox - hx) / self.board_size, -1, 1)
        obs[13] = np.clip((oy - hy) / self.board_size, -1, 1)

        obs[14] = self.us.health / 100.0
        obs[15] = self.opponent.health / 100.0
        obs[16] = self.us.length / (self.board_size * self.board_size)
        obs[17] = self.opponent.length / (self.board_size * self.board_size)

        mid = self.board_size / 2
        obs[18] = np.clip((mid - hx) / self.board_size, -1, 1)
        obs[19] = np.clip((mid - hy) / self.board_size, -1, 1)

        opp_would_lose = self.opponent.length < self.us.length
        for i, move in enumerate([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]):
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            opp_could_reach = abs(nx - ox) + abs(ny - oy) == 1
            obs[20 + i] = 1.0 if (opp_could_reach and not opp_would_lose) else 0.0

        return obs

    @staticmethod
    def _random_safe_opponent_policy(env, opponent, other):
        hx, hy = opponent.head
        occupied = set(opponent.body) | set(other.body)
        safe_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < env.board_size and 0 <= ny < env.board_size and (nx, ny) not in occupied:
                safe_moves.append(move)
        return random.choice(safe_moves) if safe_moves else random.choice(list(Move))
    @staticmethod
    def _food_seeking_opponent_policy(env, opponent, other):
        hx, hy = opponent.head
        occupied = set(opponent.body) | set(other.body)
        safe_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < env.board_size and 0 <= ny < env.board_size and (nx, ny) not in occupied:
                safe_moves.append(move)
        if not safe_moves:
            return random.choice(list(Move))
        if not env.food:
            return random.choice(safe_moves)
        fx, fy = min(env.food, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))
        return min(safe_moves, key=lambda m: abs((hx + _DELTAS[m][0]) - fx) + abs((hy + _DELTAS[m][1]) - fy))
    @staticmethod
    def _flood_fill_opponent_policy(env, opponent, other):
        hx, hy = opponent.head
        occupied = set(opponent.body) | set(other.body)
        safe_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = _DELTAS[move]
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < env.board_size and 0 <= ny < env.board_size and (nx, ny) not in occupied:
                safe_moves.append(move)
        if not safe_moves:
            return random.choice(list(Move))
        floodfill_occupied = (set(opponent.body[:-1]) | set(other.body[:-1])) - {(hx, hy)}
        def space_then_food(m):
            nx, ny = hx + _DELTAS[m][0], hy + _DELTAS[m][1]
            space = flood_fill_size((nx, ny), floodfill_occupied, env.board_size, cap=30)
            if env.food:
                fx, fy = min(env.food, key=lambda f: abs(f[0] - nx) + abs(f[1] - ny))
                food_dist = abs(nx - fx) + abs(ny - fy)
            else:
                food_dist = 0
            return (space, -food_dist)
        return max(safe_moves, key=space_then_food)

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
