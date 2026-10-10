import numpy as np
from stable_baselines3 import PPO
from train_ppo import TcpCongestionEnv


MODEL = "ppo_congestion_controller"

model = PPO.load(MODEL)

env = TcpCongestionEnv()

action_names = {
    0: "Decrease",
    1: "Maintain",
    2: "Increase"
}

action_counts = {
    0: 0,
    1: 0,
    2: 0
}

episodes = 100
total_reward = 0.0
total_steps = 0

print()
print("==========================================")
print("SEQUENTIAL PPO EVALUATION")
print("==========================================")

for episode in range(episodes):

    obs, _ = env.reset(
        seed=1000 + episode
    )

    episode_reward = 0.0

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        action = int(action)

        action_counts[action] += 1

        obs, reward, terminated, truncated, _ = env.step(
            action
        )

        episode_reward += reward
        total_steps += 1

        if terminated or truncated:
            break

    total_reward += episode_reward


average_reward = total_reward / episodes

print()
print(f"Episodes: {episodes}")
print(f"Steps: {total_steps}")

print()
print("Action distribution:")

for action in [0, 1, 2]:

    percentage = (
        action_counts[action]
        / total_steps
        * 100
    )

    print(
        f"{action} ({action_names[action]}): "
        f"{action_counts[action]} "
        f"({percentage:.2f}%)"
    )

print()
print(
    f"Average episode reward: "
    f"{average_reward:.4f}"
)

print()
print("==========================================")
