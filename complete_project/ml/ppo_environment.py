import csv
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


BASE = Path(__file__).resolve().parent

DATA = BASE / "tcp_ml_dataset.csv"
ML_MODEL = BASE / "congestion_model.pkl"


with open(ML_MODEL, "rb") as f:
    ml_model = pickle.load(f)


with open(DATA, newline="") as f:
    samples = list(csv.DictReader(f))


STATE = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}


class TcpCongestionEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        self.sample = None
        self.sample_index = 0
        self.cwnd_factor = 1.0
        self.steps = 0
        self.max_steps = 20


    def select_network_state(self):

        self.sample_index = int(
            self.np_random.integers(
                0,
                len(samples)
            )
        )

        self.sample = samples[self.sample_index]


    def predict_congestion(self):

        features = [[
            float(self.sample["bandwidth_mbps"]),
            float(self.sample["configured_delay_ms"]),
            float(self.sample["cwnd_bytes"]) * self.cwnd_factor,
            float(self.sample["total_rx_bytes"]),
            float(self.sample["instant_throughput_mbps"]),
            float(self.sample["flow_mean_delay_ms"]),
            float(self.sample["flow_loss_rate_percent"]),
            float(self.sample["flow_lost_packets"])
        ]]

        prediction = ml_model.predict(features)[0]

        return STATE[prediction]


    def get_observation(self):

        congestion = self.predict_congestion()

        return np.array([
            min(
                float(self.sample["bandwidth_mbps"]) / 10.0,
                1.0
            ),
            min(
                float(self.sample["configured_delay_ms"]) / 150.0,
                1.0
            ),
            (self.cwnd_factor - 0.5) / 1.0,
            min(
                float(self.sample["instant_throughput_mbps"]) / 10.0,
                1.0
            ),
            min(
                float(self.sample["flow_mean_delay_ms"]) / 500.0,
                1.0
            ),
            min(
                float(self.sample["flow_loss_rate_percent"]) / 100.0,
                1.0
            ),
            congestion / 2.0
        ], dtype=np.float32)


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.select_network_state()

        self.cwnd_factor = 1.0
        self.steps = 0

        return self.get_observation(), {}


    def step(self, action):

        bandwidth = float(
            self.sample["bandwidth_mbps"]
        )

        base_throughput = float(
            self.sample["instant_throughput_mbps"]
        )

        base_delay = float(
            self.sample["flow_mean_delay_ms"]
        )

        base_loss = float(
            self.sample["flow_loss_rate_percent"]
        )

        congestion = self.sample["congestion_label"]

        old_cwnd = self.cwnd_factor


        if action == 0:
            self.cwnd_factor *= 0.80

        elif action == 1:
            self.cwnd_factor *= 1.00

        else:
            self.cwnd_factor *= 1.20


        self.cwnd_factor = float(
            np.clip(
                self.cwnd_factor,
                0.5,
                1.5
            )
        )


        cwnd_change = self.cwnd_factor - old_cwnd


        if congestion == "Low":

            throughput = (
                base_throughput
                * (1.0 + 0.35 * cwnd_change)
            )

            delay = (
                base_delay
                * (1.0 + 0.10 * max(cwnd_change, 0))
            )

            loss = base_loss


        elif congestion == "Medium":

            throughput = (
                base_throughput
                * (
                    1.0
                    + 0.10 * min(cwnd_change, 0)
                    - 0.10 * max(cwnd_change, 0)
                )
            )

            delay = (
                base_delay
                * (1.0 + 0.40 * abs(cwnd_change))
            )

            loss = (
                base_loss
                + 1.5 * max(cwnd_change, 0)
            )


        else:

            throughput = (
                base_throughput
                * (
                    1.0
                    - 0.10 * max(cwnd_change, 0)
                )
            )

            delay = (
                base_delay
                * (
                    1.0 + 0.70 * abs(cwnd_change)
                )
            )

            loss = (
                base_loss
                + 5.0 * max(cwnd_change, 0)
            )


        throughput = min(
            max(throughput, 0.001),
            bandwidth
        )

        delay = max(delay, 1.0)

        loss = max(loss, 0.0)


        throughput_score = (
            throughput / max(bandwidth, 1.0)
        )

        delay_penalty = min(
            delay / 300.0,
            2.0
        )

        loss_penalty = min(
            loss / 10.0,
            2.0
        )


        reward = (
            4.0 * throughput_score
            - 1.5 * delay_penalty
            - 2.0 * loss_penalty
        )


        if congestion == "Low":

            if action == 2:
                reward += 1.0

            elif action == 0:
                reward -= 1.0


        elif congestion == "Medium":

            if action == 1:
                reward += 1.0

            elif action == 2:
                reward -= 0.5


        else:

            if action == 0:
                reward += 1.0

            elif action == 2:
                reward -= 1.5


        self.steps += 1

        if self.steps % 4 == 0:
            self.select_network_state()

        terminated = self.steps >= self.max_steps

        return (
            self.get_observation(),
            float(reward),
            terminated,
            False,
            {}
        )
