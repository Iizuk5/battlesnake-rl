"""
Train a PPO agent to play Battlesnake against the built-in heuristic opponent.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.battlesnake_env import BattlesnakeEnv


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, print_every=50):
        super().__init__()
        self.print_every = print_every
        self.episode_count = 0
        self.wins = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                self.episode_count += 1
                ep_reward = info.get("episode", {}).get("r", 0)
                if ep_reward > 0:
                    self.wins += 1
                if self.episode_count % self.print_every == 0:
                    win_rate = self.wins / self.episode_count * 100
                    print(
                        f"  episode {self.episode_count} — "
                        f"win rate so far: {win_rate:.1f}% ({self.wins}/{self.episode_count})",
                        flush=True,
                    )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--board-size", type=int, default=11)
    parser.add_argument("--out", type=str, default="models/ppo_battlesnake.zip")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    args = parser.parse_args()

    env = BattlesnakeEnv(board_size=args.board_size)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    checkpoint_dir = os.path.join(os.path.dirname(args.out) or ".", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=checkpoint_dir,
        name_prefix="ppo_battlesnake",
    )
    callbacks = CallbackList([EpisodeStatsCallback(), checkpoint_callback])

    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
