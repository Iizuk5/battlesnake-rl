"""
Evaluate a trained model's win rate, and optionally watch a battle play out
turn-by-turn in the terminal.

Usage:
  python scripts/evaluate.py --model models/ppo_battlesnake.zip --n-episodes 200
  python scripts/evaluate.py --model models/ppo_battlesnake.zip --watch
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO

from env.battlesnake_env import BattlesnakeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--board-size", type=int, default=11)
    parser.add_argument("--watch", action="store_true", help="Print one battle turn-by-turn instead of running a batch")
    args = parser.parse_args()

    env = BattlesnakeEnv(board_size=args.board_size)
    model = PPO.load(args.model)

    if args.watch:
        obs, info = env.reset()
        print(env.render())
        print()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            time.sleep(0.3)
            print(env.render())
            print(f"reward: {reward:.2f}\n")
        result = "WIN" if env.opponent.alive is False and env.us.alive else (
            "LOSS" if not env.us.alive else "DRAW/TIMEOUT"
        )
        print(f"Result: {result}")
        return

    wins = 0
    losses = 0
    draws = 0
    for _ in range(args.n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        if not env.us.alive and not env.opponent.alive:
            draws += 1
        elif not env.us.alive:
            losses += 1
        elif not env.opponent.alive:
            wins += 1
        else:
            draws += 1  # truncated at max_turns with both alive

    total = wins + losses + draws
    print(f"\nResults over {total} episodes:")
    print(f"  Wins:   {wins} ({100 * wins / total:.1f}%)")
    print(f"  Losses: {losses} ({100 * losses / total:.1f}%)")
    print(f"  Draws:  {draws} ({100 * draws / total:.1f}%)")


if __name__ == "__main__":
    main()
