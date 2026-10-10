import csv
import numpy as np
from stable_baselines3 import PPO

MODEL = "ppo_congestion_controller"
DATA = "tcp_ml_dataset.csv"

model = PPO.load(MODEL)

state_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

desired_action = {
    "Low": 2,
    "Medium": 1,
    "High": 0
}

action_names = {
    0: "Decrease",
    1: "Maintain",
    2: "Increase"
}

rows = []

with open(DATA, newline="") as f:
    rows = list(csv.DictReader(f))

actions = {0: 0, 1: 0, 2: 0}
correct = 0

for row in rows:

    # First use the trained RF model indirectly through
    # the congestion label already present in the dataset.
    congestion = row["congestion_label"]

    observation = np.array([
        float(row["bandwidth_mbps"]),
        float(row["configured_delay_ms"]),
        float(row["cwnd_bytes"]),
        float(row["instant_throughput_mbps"]),
        float(row["flow_mean_delay_ms"]),
        float(row["flow_loss_rate_percent"]),
        float(state_map[congestion])
    ], dtype=np.float32)

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    actions[action] += 1

    if action == desired_action[congestion]:
        correct += 1

accuracy = correct / len(rows)

print("\n===== PPO ACTION EVALUATION =====")
print(f"Samples: {len(rows)}")

print("\nAction distribution:")
for action in [0, 1, 2]:
    print(
        f"{action} ({action_names[action]}): "
        f"{actions[action]}"
    )

print(
    f"\nDesired-action matches: "
    f"{correct}/{len(rows)}"
)

print(f"Action accuracy: {accuracy:.4f}")
print(f"Action accuracy (%): {accuracy * 100:.2f}%")
