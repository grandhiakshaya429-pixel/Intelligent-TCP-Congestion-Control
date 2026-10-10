from pathlib import Path

import torch
import torch.nn as nn
from stable_baselines3 import PPO


BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "ppo_congestion_controller"
OUTPUT_PATH = BASE / "ppo_policy.onnx"


# Load the existing trained model.
model = PPO.load(MODEL_PATH, device="cpu")

policy = model.policy
policy.set_training_mode(False)


class PPOActor(nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

    def forward(self, observation):
        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        return torch.argmax(logits, dim=1)


actor = PPOActor(policy).eval()

dummy_input = torch.zeros((1, 7), dtype=torch.float32)

torch.onnx.export(
    actor,
    dummy_input,
    str(OUTPUT_PATH),
    input_names=["observation"],
    output_names=["action"],
    opset_version=17,
    dynamo=False,
)

print("PPO ONNX export complete.")
print(f"Output: {OUTPUT_PATH}")
