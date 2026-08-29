# Battlesnake RL Bot

A self-contained reinforcement learning project for [Battlesnake](https://play.battlesnake.com) —
no external game-connection library, no server dependency for training. The game rules are
implemented directly in `env/battlesnake_env.py`, so training runs instantly and locally.

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

That's it — no separate game server needed for training (unlike the Pokémon Showdown
approach), since the environment simulates the game rules itself.

## 2. Project layout

```
battlesnake-rl/
├── env/
│   └── battlesnake_env.py   # Game rules + Gymnasium environment (self-contained)
├── scripts/
│   ├── train.py               # Training loop (Stable-Baselines3 PPO)
│   └── evaluate.py            # Win-rate evaluation, or --watch a battle turn-by-turn
├── server/
│   └── app.py                  # Flask server implementing the real Battlesnake API,
│                                 for deploying your trained model to actual games
├── models/                     # Trained models saved here
└── requirements.txt
```

## 3. Quickstart

```bash
# Train (should run fast — no network calls, pure local simulation)
python scripts/train.py --timesteps 200000

# Watch a trained agent play one battle turn-by-turn in the terminal
python scripts/evaluate.py --model models/ppo_battlesnake.zip --watch

# Get a win-rate number over many episodes
python scripts/evaluate.py --model models/ppo_battlesnake.zip --n-episodes 200
```

Since there's no server round-trip, training should be dramatically faster than the
Showdown/poke-env approach — expect thousands of steps per second on modest hardware,
not seconds per step. 200,000 timesteps should take minutes, not hours.

## 4. Playing real games (optional, once your agent is decent)

Battlesnake's actual competitive API is simple and has been stable for years — a snake is
just an HTTP server responding to `/start`, `/move`, and `/end` requests with JSON.
`server/app.py` implements this, loading your trained model to answer `/move` requests.

```bash
python server/app.py --model models/ppo_battlesnake.zip --port 8080
```

To have real games reach your local server, you'll need to expose it publicly — the
simplest option is a tunnel tool like `ngrok`:
```bash
ngrok http 8080
```
Then register the ngrok URL as your snake's URL at https://play.battlesnake.com — full
walkthrough at https://docs.battlesnake.com/quickstart.

**Important**: `board_to_obs()` in `server/app.py` must encode the board state into the
*exact* same 20-value observation format used during training (see `_get_obs()` in
`env/battlesnake_env.py`). If you change one, change the other to match — I've kept them
in sync in this scaffold, but it's the one place a future edit is easy to break silently
(the model will still run, just make worse decisions, since observations won't match what
it was trained on).

## 5. Design notes / what to tune next

- **Opponent**: training uses a simple heuristic opponent (`_default_opponent_policy` in
  `battlesnake_env.py`) that just avoids obviously-fatal moves. Once your agent reliably
  beats it, try self-play (have the opponent also be a copy of the trained policy,
  updated periodically) for a stronger final agent.
- **Observation space**: currently 20 values — 4 danger flags, food direction, opponent
  direction, health/length for both snakes, and direction to board center. This is
  intentionally minimal. Next steps: encode a small local grid patch around the head
  instead of just 4 binary danger flags (lets the agent "see" further ahead), or add
  awareness of the opponent's likely next move.
- **Reward shaping**: currently sparse-ish (+10 win / -10 loss / +0.01 per-turn survival).
  Consider adding a small reward for eating food, or a penalty for shrinking the available
  safe space (getting boxed in), if training plateaus.
- **Board size**: defaults to 11x11 (the standard competitive size). Training on a smaller
  board first (e.g. 7x7) can speed up early learning before moving to full size.
