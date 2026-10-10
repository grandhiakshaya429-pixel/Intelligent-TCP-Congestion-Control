Intelligent TCP Congestion Control Using Hybrid ML and Reinforcement Learning
An experimental TCP congestion-control prototype implemented in ns-3 that combines a Random Forest (RF) congestion-state classifier with a Proximal Policy Optimization (PPO) policy. The trained models are exported to ONNX and called from C++ using ONNX Runtime.
> **Research status:** This is a prototype. Recorded experiments show mixed throughput results and lower measured mean delay in the six tested scenarios. These results do not establish universal improvement over TCP Reno.
Contents
Overview
Architecture
Technology stack
Repository layout
Requirements
Run the hybrid experiment
Results and graphs
Recorded observations
Important limitations
Reproducibility notes
Overview
The prototype uses two models:
Random Forest classifier: classifies observed network conditions as `Low`, `Medium`, or `High`.
PPO policy: selects `Decrease`, `Maintain`, or `Increase` at controller decision points.
The action multipliers used by the current policy are:
`Decrease`: CWND factor × 0.8
`Maintain`: no action-based scaling at that decision point
`Increase`: CWND factor × 1.2
The action factor is bounded between 0.5 and 1.5. ACK-driven growth can continue between decision points.
Architecture
ns-3 configures a TCP network scenario.
The controller gathers runtime features such as CWND, acknowledgement-derived bytes, throughput estimates, RTT, and loss-related indicators.
The RF ONNX model predicts a congestion label.
The PPO ONNX model selects an action.
The controller applies the action and logs its decisions.
FlowMonitor output is compared with a TCP Reno baseline.
Technology stack
C++ and ns-3
Python for model development and evaluation
Random Forest classifier
Proximal Policy Optimization (PPO)
ONNX model format
ONNX Runtime for C++ inference
CSV and PNG files for results
Repository layout
```text
Intelligent-TCP-Congestion-Control/
├── ml/
│   ├── congestion_model.onnx
│   ├── congestion_model.pkl
│   ├── ppo_policy.onnx
│   └── ppo_environment.py
├── ns-3/
│   ├── src/internet/model/
│   │   ├── tcp-hybrid-ml-rl.h
│   │   └── tcp-hybrid-ml-rl.cc
│   └── scratch/
│       └── tcp-hybrid-ml-rl-experiment.cc
└── results/
    ├── csv/
    │   ├── hybrid_vs_reno_comparison.csv
    │   ├── trial1_backup/
    │   ├── trial2/
    │   └── trial2_clean/
    └── graphs/
        ├── throughput_comparison.png
        ├── delay_comparison.png
        ├── throughput_change.png
        └── cwnd_comparison_10Mbps_50ms.png
```
Some additional files may exist in the working repository; this tree lists the main files used in the experiment.
Requirements
Use the environment already configured for this project:
WSL Ubuntu
The configured ns-3 build and its dependencies
Python virtual environment at `ml-venv`
The RF and PPO ONNX model files
ONNX Runtime libraries and headers available to the ns-3 build
From the project root, activate the Python environment when using Python tools:
```bash
cd ~/Intelligent-TCP-Congestion-Control
source ml-venv/bin/activate
```
Run the hybrid experiment
Run from the ns-3 directory:
```bash
cd ~/Intelligent-TCP-Congestion-Control/ns-3
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=10Mbps --delay=50ms"
```
Example scenario variations:
```bash
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=10Mbps --delay=25ms"
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=10Mbps --delay=150ms"
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=1Mbps --delay=100ms"
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=2Mbps --delay=50ms"
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=5Mbps --delay=100ms"
```
The experiment prints a completion message and writes CSV outputs according to the configured output paths in the experiment code. Check the source for the exact file names and paths if they have been changed.
Repeat-run warning
The decision logger appends to its output file. Before repeating a run that writes to the same decision-log path, move or back up the previous file, or remove only that specific output file after confirming its path. Otherwise the CSV may contain multiple runs appended together. The saved clean Trial 2 log contains 188 records; the original Trial 2 log contains 376 because two identical 188-record sequences were appended.
Results and graphs
The recorded six-scenario comparison is in:
```text
results/csv/hybrid_vs_reno_comparison.csv
```
The graphs are in `results/graphs/`:
`throughput_comparison.png`
`delay_comparison.png`
`throughput_change.png`
`cwnd_comparison_10Mbps_50ms.png`
The clean Trial 2 files are stored in `results/csv/trial2_clean/`, and the Trial 1 backups are in `results/csv/trial1_backup/`.
Recorded observations
The current comparison CSV reports:
Scenario	Throughput change vs. Reno	Mean-delay change vs. Reno
10 Mbps / 150 ms	-5.38%	-0.02%
10 Mbps / 25 ms	+0.17%	-5.68%
10 Mbps / 50 ms	-1.11%	-0.15%
1 Mbps / 100 ms	+5.63%	-30.13%
2 Mbps / 50 ms	-0.58%	-7.15%
5 Mbps / 100 ms	-2.80%	-0.44%
Positive throughput change means Hybrid throughput was higher. Negative mean-delay change means Hybrid mean delay was lower. The values are observations from the recorded runs, not statistical significance claims.
Important limitations
PPO was trained in a synthetic environment, not directly through interaction with the evaluated ns-3 network.
Some runtime features are proxies rather than exact matches to the training features. For example, latest RTT is used as a delay proxy, while the RF training feature is flow mean delay.
The runtime loss-event counter is not the same as FlowMonitor's packet-loss count.
In the inspected 10 Mbps / 50 ms decision trace, PPO selected `Maintain` 175 times, `Increase` 13 times, and `Decrease` 0 times. The RF output was `Low` 15 times and `Medium` 173 times.
The six-scenario comparison is based on recorded runs and does not establish statistical significance or generalization to real networks.
FlowMonitor-reported `LostPackets` should not be interpreted as `TxPackets - RxPackets`.
Reproducibility notes
Two runs of the Hybrid 10 Mbps / 50 ms scenario produced identical recorded FlowMonitor values and identical CWND traces in the current setup. This is a reproducibility check for that configuration, not a substitute for multiple independent seeds and statistical analysis.
Suggested next improvements
Align runtime features and preprocessing with the model's training feature definitions.
Train PPO in the same ns-3 environment or a carefully validated simulator-matched environment.
Investigate why the policy rarely chooses `Decrease`.
Run multiple independent seeds for each scenario and report means and variability.
Test competing TCP flows, controlled queue bottlenecks, and explicit packet-error scenarios.
Evaluate fairness as well as throughput and delay.