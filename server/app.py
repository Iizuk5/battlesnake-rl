"""
Flask server implementing Battlesnake's HTTP API, using a trained PPO model
to choose moves.
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from flask import Flask, request, jsonify
from stable_baselines3 import PPO

app = Flask(__name__)

MOVE_NAMES = ["up", "down", "left", "right"]


def load_model(model_path: str):
    global model
    model = PPO.load(model_path)
    return model


_default_model_path = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "ppo_battlesnake.zip"),
)
model = None
if os.path.exists(_default_model_path):
    load_model(_default_model_path)


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

    obs = np.zeros(20, dtype=np.float32)
    hx, hy = my_body[0]
    occupied = set(my_body) | set(opp_body)

    deltas = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    for i, direction in enumerate(["up", "down", "left", "right"]):
        dx, dy = deltas[direction]
        nx, ny = hx + dx, hy + dy
        danger = not (0 <= nx < board_size and 0 <= ny < board_size) or (nx, ny) in occupied
        obs[i] = 1.0 if danger else 0.0

    if food:
        fx, fy = min(food, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))
        obs[4] = np.clip((fx - hx) / board_size, -1, 1)
        obs[5] = np.clip((fy - hy) / board_size, -1, 1)

    ox, oy = opp_body[0]
    obs[6] = np.clip((ox - hx) / board_size, -1, 1)
    obs[7] = np.clip((oy - hy) / board_size, -1, 1)

    obs[8] = you["health"] / 100.0
    obs[9] = opp_health / 100.0
    obs[10] = len(my_body) / (board_size * board_size)
    obs[11] = len(opp_body) / (board_size * board_size)

    mid = board_size / 2
    obs[12] = np.clip((mid - hx) / board_size, -1, 1)
    obs[13] = np.clip((mid - hy) / board_size, -1, 1)

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
    return jsonify({"move": MOVE_NAMES[int(action)]})


@app.route("/end", methods=["POST"])
def end():
    return jsonify({})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_battlesnake.zip")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    load_model(args.model)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
