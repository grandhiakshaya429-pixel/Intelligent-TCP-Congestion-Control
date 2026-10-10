from pathlib import Path

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO


BASE = Path(__file__).resolve().parent

ppo = PPO.load(
    BASE / "ppo_congestion_controller",
    device="cpu"
)

session = ort.InferenceSession(
    str(BASE / "ppo_policy.onnx"),
    providers=["CPUExecutionProvider"]
)

rng = np.random.default_rng(42)

test_observations = rng.uniform(
    0.0, 1.0, size=(1000, 7)
).astype(np.float32)

matches = 0

for observation in test_observations:

    original_action, _ = ppo.predict(
        observation,
        deterministic=True
    )

    onnx_action = session.run(
        ["action"],
        {"observation": observation.reshape(1, 7)}
    )[0]

    if int(original_action) == int(onnx_action[0]):
        matches += 1

agreement = matches / len(test_observations) * 100

print("\n===== ONNX VERIFICATION =====")
print(f"Test observations: {len(test_observations)}")
print(f"Matching actions: {matches}")
print(f"Action agreement: {agreement:.2f}%")

if matches == len(test_observations):
    print("PASS: All actions match.")
else:
    print("FAIL: Investigate before C++ integration.")
