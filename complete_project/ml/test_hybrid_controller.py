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

action_names = {
    0: "DECREASE CWND",
    1: "MAINTAIN CWND",
    2: "INCREASE CWND"
}

label_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

samples = []

with open(DATA, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        samples.append(row)


print("\n==========================================")
print("HYBRID ML + PPO CONTROLLER TEST")
print("==========================================\n")

correct = 0
total = 0

# Test one representative sample from each congestion class
tested_labels = set()

for row in samples:

    actual_label = row["congestion_label"]

    if actual_label in tested_labels:
        continue

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

    predicted_label = ml_model.predict(features)[0]

    observation = np.array([
        float(row["bandwidth_mbps"]),
        float(row["configured_delay_ms"]),
        float(row["cwnd_bytes"]),
        float(row["instant_throughput_mbps"]),
        float(row["flow_mean_delay_ms"]),
        float(row["flow_loss_rate_percent"]),
        float(label_map[predicted_label])
    ], dtype=np.float32)

    action, _ = ppo_model.predict(
        observation,
        deterministic=True
    )

    action = int(action)

    print(f"Actual congestion : {actual_label}")
    print(f"ML prediction     : {predicted_label}")
    print(f"PPO action        : {action_names[action]}")
    print("------------------------------------------")

    expected_action = {
        "Low": 2,
        "Medium": 1,
        "High": 0
    }[actual_label]

    if action == expected_action:
        correct += 1

    total += 1
    tested_labels.add(actual_label)

    if len(tested_labels) == 3:
        break


print(f"\nCorrect decisions: {correct}/{total}")

if total > 0:
    accuracy = correct / total * 100
    print(f"Decision accuracy: {accuracy:.2f}%")

print("\n==========================================")
print("HYBRID CONTROLLER TEST COMPLETE")
print("==========================================")
