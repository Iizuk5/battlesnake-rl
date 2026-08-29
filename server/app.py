"""
Flask server implementing Battlesnake's HTTP API, using a trained PPO model
to choose moves.
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from flask import Flask, request, jsonify
from stable_baselines3 import PPO

from env.battlesnake_env import BattlesnakeEnv, flood_fill_size

app = Flask(__name__)

MOVE_NAMES = ["up", "down", "left", "right"]


def load_model(weights_path: str):
    global model
    dummy_env = BattlesnakeEnv()
    model = PPO("MlpPolicy", dummy_env, device="cpu")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(state_dict)
    model.policy.eval()
    return model


_default_weights_path = os.environ.get(
    "WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "policy_weights.pth"),
)
model = None
if os.path.exists(_default_weights_path):
    print(f"[diagnostic] weights file: {_default_weights_path}", flush=True)
    print(f"[diagnostic] size: {os.path.getsize(_default_weights_path)} bytes", flush=True)
    load_model(_default_weights_path)


def board_to_obs(data):
    board = data["board"]
    board_size = board["width"]
    you = data["you"]

    my_body = [(seg["x"], seg["y"]) for seg in you["body"]]
    opponents = [s for s in board["snakes"] if s["id"] != you["id"]]
    if opponents:
        opp = min(
            opponents,
            key=lambda s: abs(s["head"]["x"] - you["head"]["x"]) + abs(s["head"]["y"] - you["head"]["y"]),
        )
        opp_body = [(seg["x"], seg["y"]) for seg in opp["body"]]
        opp_health = opp["health"]
    else:
        opp_body = [(-1, -1)]
        opp_health = 0

    food = [(f["x"], f["y"]) for f in board["food"]]

    obs = np.zeros(24, dtype=np.float32)
    hx, hy = my_body[0]
    our_occupied = set(my_body) | set(opp_body)
    floodfill_occupied = (set(my_body[:-1]) | set(opp_body[:-1])) - {(hx, hy)}

    deltas = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    for i, direction in enumerate(["up", "down", "left", "right"]):
        dx, dy = deltas[direction]
        nx, ny = hx + dx, hy + dy
        danger = not (0 <= nx < board_size and 0 <= ny < board_size) or (nx, ny) in our_occupied
        obs[i] = 1.0 if danger else 0.0

        space = flood_fill_size((nx, ny), floodfill_occupied, board_size, cap=30)
        obs[4 + i] = min(space / 30.0, 1.0)

    food_sorted = sorted(food, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))
    if len(food_sorted) >= 1:
        fx, fy = food_sorted[0]
        obs[8] = np.clip((fx - hx) / board_size, -1, 1)
        obs[9] = np.clip((fy - hy) / board_size, -1, 1)
    if len(food_sorted) >= 2:
        fx2, fy2 = food_sorted[1]
        obs[10] = np.clip((fx2 - hx) / board_size, -1, 1)
        obs[11] = np.clip((fy2 - hy) / board_size, -1, 1)

    ox, oy = opp_body[0]
    obs[12] = np.clip((ox - hx) / board_size, -1, 1)
    obs[13] = np.clip((oy - hy) / board_size, -1, 1)

    obs[14] = you["health"] / 100.0
    obs[15] = opp_health / 100.0
    obs[16] = len(my_body) / (board_size * board_size)
    obs[17] = len(opp_body) / (board_size * board_size)

    mid = board_size / 2
    obs[18] = np.clip((mid - hx) / board_size, -1, 1)
    obs[19] = np.clip((mid - hy) / board_size, -1, 1)

    opp_would_lose = len(opp_body) < len(my_body)
    for i, direction in enumerate(["up", "down", "left", "right"]):
        dx, dy = deltas[direction]
        nx, ny = hx + dx, hy + dy
        opp_could_reach = abs(nx - ox) + abs(ny - oy) == 1
        obs[20 + i] = 1.0 if (opp_could_reach and not opp_would_lose) else 0.0

    return obs


@app.route("/", methods=["GET"])
def info():
    return jsonify(
        {
            "apiversion": "1",
            "author": "your-username",
            "color": "#5c8dff",
            "head": "default",
            "tail": "default",
        }
    )


@app.route("/start", methods=["POST"])
def start():
    return jsonify({})


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    obs = board_to_obs(data)
    action, _ = model.predict(obs, deterministic=True)
    print(f"[move-debug] raw request: {data}", flush=True)
    print(f"[move-debug] computed obs: {obs.tolist()}", flush=True)
    print(f"[move-debug] chosen action: {int(action)} ({MOVE_NAMES[int(action)]})", flush=True)
    return jsonify({"move": MOVE_NAMES[int(action)]})


@app.route("/end", methods=["POST"])
def end():
    return jsonify({})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="models/policy_weights.pth")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    load_model(args.weights)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
