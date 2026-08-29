"""
Export a trained PPO model's policy weights in PyTorch's older, non-zip
serialization format.
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_battlesnake.zip")
    parser.add_argument("--out", type=str, default="models/policy_weights.pth")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    torch.save(
        model.policy.state_dict(),
        args.out,
        _use_new_zipfile_serialization=False,
    )
    print(f"Exported policy weights to {args.out}")
    print(f"Size: {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
