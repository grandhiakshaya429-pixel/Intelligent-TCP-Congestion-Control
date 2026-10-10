from pathlib import Path

from stable_baselines3 import PPO

from ppo_environment import TcpCongestionEnv


BASE = Path(__file__).resolve().parent

PPO_MODEL = BASE / "ppo_congestion_controller"


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

env = TcpCongestionEnv()


# ============================================================
# PPO MODEL
# ============================================================

model = PPO(
    "MlpPolicy",
    env,

    verbose=1,

    learning_rate=0.0003,

    n_steps=1024,

    batch_size=64,

    gamma=0.98,

    gae_lambda=0.95,

    ent_coef=0.03,

    clip_range=0.2,

    seed=42
)


print()
print("==========================================")
print("TRAINING SEQUENTIAL PPO CONTROLLER")
print("==========================================")
print()


model.learn(
    total_timesteps=150000
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(PPO_MODEL)


print()
print("==========================================")
print("SEQUENTIAL PPO TRAINING COMPLETE")
print("==========================================")
print()
print(f"Model saved to: {PPO_MODEL}")
print()
