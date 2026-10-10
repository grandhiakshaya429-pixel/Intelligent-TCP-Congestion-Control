import csv
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

BASE = Path(__file__).resolve().parent

DATA = BASE / "tcp_ml_dataset.csv"
ML_MODEL = BASE / "congestion_model.pkl"
PPO_MODEL = BASE / "ppo_congestion_controller"

with open(ML_MODEL, "rb") as f:
    ml_model = pickle.load(f)

ppo_model = PPO.load(PPO_MODEL)

expected = {
    "Low": 2,
    "Medium": 1,
    "High": 0
}

counts = {
    "Low": 0,
    "Medium": 0,
    "High": 0
}

correct = {
    "Low": 0,
    "Medium": 0,
    "High": 0
}

total = 0

with open(DATA, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        features = [[
            float(row["bandwidth_mbps"]),
            float(row["configured_delay_ms"]),
            float(row["cwnd_bytes"]),
            float(row["total_rx_bytes"]),
            float(row["instant_throughput_mbps"]),
            float(row["flow_mean_delay_ms"]),
            float(row["flow_loss_rate_percent"]),
            float(row["flow_lost_packets"])
        ]]

        prediction = ml_model.predict(features)[0]

        state = {
            "Low": 0,
            "Medium": 1,
            "High": 2
        }[prediction]

        observation = np.array([
            float(row["bandwidth_mbps"]),
            float(row["configured_delay_ms"]),
            float(row["cwnd_bytes"]),
            float(row["instant_throughput_mbps"]),
            float(row["flow_mean_delay_ms"]),
            float(row["flow_loss_rate_percent"]),
            float(state)
        ], dtype=np.float32)

        action, _ = ppo_model.predict(
            observation,
            deterministic=True
        )

        action = int(action)

        actual = row["congestion_label"]

        counts[actual] += 1

        if action == expected[actual]:
            correct[actual] += 1

        total += 1

print("\n==========================================")
print("HYBRID ML + PPO FULL EVALUATION")
print("==========================================")

print(f"\nTotal samples: {total}")

overall_correct = sum(correct.values())

print(f"Correct decisions: {overall_correct}/{total}")

print(
    f"Overall decision accuracy: "
    f"{overall_correct / total * 100:.2f}%"
)

print("\nPer-class results:")

for label in ["Low", "Medium", "High"]:

    accuracy = (
        correct[label] / counts[label] * 100
        if counts[label] else 0
    )

    print(
        f"{label:8s}: "
        f"{correct[label]}/{counts[label]} "
        f"({accuracy:.2f}%)"
    )

print("\n==========================================")
print("EVALUATION COMPLETE")
print("==========================================")
