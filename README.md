# Intelligent TCP Congestion Control Using Hybrid ML and Reinforcement Learning

An experimental TCP congestion-control project that combines **Machine Learning (Random Forest)**, **Reinforcement Learning (PPO)**, and **ns-3 network simulation** to investigate adaptive congestion-control strategies.

The project includes a hybrid TCP controller, trained ML model, simulation experiments, CSV results, comparison graphs, and a website dashboard for visualizing performance.

> **Project status:** Experimental prototype. Results are based on the network configurations tested in this project and do not establish that the hybrid controller outperforms TCP Reno in every scenario.

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Objectives](#-objectives)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Technologies Used](#-technologies-used)
* [Project Structure](#-project-structure)
* [Website Dashboard](#-website-dashboard)
* [Installation and Execution](#-installation-and-execution)
* [Experimental Results](#-experimental-results)
* [Understanding the Results](#-understanding-the-results)
* [Limitations](#-limitations)
* [Future Enhancements](#-future-enhancements)

---

## 📖 Overview

TCP congestion control regulates the amount of data a sender transmits to avoid overwhelming the network.

Traditional algorithms such as TCP Reno adjust their behavior based on network feedback. This project explores a hybrid approach that combines ML-based congestion classification with RL-based actions to adjust the TCP congestion window.

The system uses ns-3 to simulate network conditions and evaluate the hybrid controller against a TCP Reno baseline.

Performance is assessed using throughput, mean delay, congestion-window behavior, controller decisions, and FlowMonitor statistics.

---

## 🎯 Objectives

* Investigate ML-assisted TCP congestion control.
* Classify network conditions using a Random Forest model.
* Integrate a PPO-based policy that selects congestion-window actions.
* Implement and evaluate the controller in ns-3.
* Compare the hybrid approach with TCP Reno across multiple network scenarios.
* Visualize experimental results using graphs and a website dashboard.

---

## ✨ Key Features

* **Random Forest Classification:** Predicts congestion categories from network metrics.
* **PPO-Based Actions:** Selects Decrease, Maintain, or Increase actions.
* **ns-3 Simulation:** Evaluates network performance under different bandwidth and delay configurations.
* **TCP Reno Baseline:** Provides a comparison against a traditional congestion-control algorithm.
* **Performance Analysis:** Examines throughput, delay, and congestion-window behavior.
* **CSV Logging:** Stores experimental measurements and controller decisions.
* **Website Dashboard:** Presents saved results and comparison graphs.

---

## 🏗️ System Architecture

```text
Network Configuration
        |
        v
     ns-3 Network Simulation
        |
        v
   Collect Network Metrics
        |
        +-----------------------+
        |                       |
        v                       v
 Random Forest Model        PPO Policy
 (ONNX Runtime)             (RL Actions)
        |                       |
        v                       v
 Congestion Classification  Decrease / Maintain / Increase
        |                       |
        +-----------+-----------+
                    |
                    v
          Hybrid TCP Controller
                    |
                    v
         Congestion Window Updates
                    |
                    v
          FlowMonitor and CSV Logs
                    |
                    v
        Graphs and Website Dashboard
```

---

## 🛠️ Technologies Used

| Technology            | Purpose                                           |
| --------------------- | ------------------------------------------------- |
| C++                   | TCP controller and ns-3 experiment implementation |
| Python                | ML/RL scripts and data analysis                   |
| ns-3                  | Network simulation                                |
| Random Forest         | Congestion classification                         |
| ONNX Runtime          | ML inference in the C++ controller                |
| PPO                   | Reinforcement-learning policy                     |
| HTML, CSS, JavaScript | Website dashboard                                 |
| CSV                   | Experimental results and decision logs            |
| Git and GitHub        | Version control and project hosting               |

---

## 📁 Project Structure

```text
Intelligent-TCP-Congestion-Control/
│
├── README.md
│
└── complete_project/
    ├── ml/
    │   ├── congestion_model.onnx
    │   ├── congestion_model.pkl
    │   ├── tcp_ml_dataset.csv
    │   └── PPO environment and supporting scripts
    │
    ├── ns-3/
    │   ├── src/
    │   │   └── internet/model/
    │   │       ├── tcp-hybrid-ml-rl.h
    │   │       └── tcp-hybrid-ml-rl.cc
    │   │
    │   └── scratch/
    │       └── tcp-hybrid-ml-rl-experiment.cc
    │
    ├── results/
    │   ├── csv/
    │   └── graphs/
    │
    └── website/
        ├── index.html
        ├── CSS files
        ├── JavaScript files
        └── supporting assets
```

*Note: The exact filenames may vary depending on the files included in the current repository.*

---

## 🌐 Website Dashboard

The project includes a static website dashboard for presenting the saved simulation results.

### Dashboard Features

* Project overview and objectives.
* System architecture explanation.
* Throughput comparison between TCP Reno and Hybrid ML-RL.
* Mean-delay comparison charts.
* Congestion-window graphs.
* Links to experiment CSV files.
* Controller insights and result summaries.

### Run the Website Locally

**Requirements:** Python 3.

Open PowerShell in the repository directory and run:

```powershell
cd complete_project/website
python -m http.server 8000
```

Open the following address in your browser:

**http://localhost:8000**

Keep the terminal open while viewing the dashboard. Press `Ctrl+C` to stop the server.

> The dashboard displays saved experimental results. It does not launch a new ns-3 simulation directly from the browser.

### Public Website Hosting

The website source code is stored in this repository. To make it accessible through a public URL, configure GitHub Pages with a workflow that publishes the contents of `complete_project/website`.

Add the public website URL here once deployment has been completed and verified.

**Live Dashboard:** Not yet configured.

---

## ⚙️ Installation and Execution

### Requirements

* Windows with WSL/Ubuntu or a compatible Linux environment.
* Git.
* Python 3.
* A compatible ns-3 build environment.
* ONNX Runtime development headers and library.
* The project's trained model and supporting files.

### Step 1: Clone the Repository

```bash
git clone https://github.com/grandhiakshaya429-pixel/Intelligent-TCP-Congestion-Control.git
cd Intelligent-TCP-Congestion-Control
```

### Step 2: Navigate to ns-3

```bash
cd complete_project/ns-3
```

### Step 3: Configure the Build

```bash
./ns3 configure
```

Review the configuration output and ensure the required dependencies, particularly ONNX Runtime, are available.

### Step 4: Build the Project

```bash
./ns3 build
```

Resolve any dependency or compilation errors before proceeding.

### Step 5: Run the Hybrid Experiment

```bash
./ns3 run "scratch/tcp-hybrid-ml-rl-experiment --simulationTime=20 --bandwidth=10Mbps --delay=50ms"
```

This command assumes the experiment file and the required controller dependencies are correctly configured.

The experiment runs a 20-second simulation with a configured bandwidth of 10 Mbps and delay of 50 ms.

---

## 📊 Experimental Results

The following table summarizes the recorded results for the primary data flow in six tested scenarios.

| Scenario         | Reno Throughput (Mbps) | Hybrid Throughput (Mbps) | Throughput Change | Reno Mean Delay (ms) | Hybrid Mean Delay (ms) | Delay Change |
| ---------------- | ---------------------: | -----------------------: | ----------------: | -------------------: | ---------------------: | -----------: |
| 10 Mbps / 150 ms |                3.36057 |                  3.17984 |            -5.38% |              151.312 |                151.288 |       -0.02% |
| 10 Mbps / 25 ms  |                9.27320 |                  9.28939 |            +0.17% |               56.339 |                 53.141 |       -5.68% |
| 10 Mbps / 50 ms  |                9.20700 |                  9.10505 |            -1.11% |               65.710 |                 65.608 |       -0.15% |
| 1 Mbps / 100 ms  |                0.84621 |                  0.89383 |            +5.63% |              359.698 |                251.312 |      -30.13% |
| 2 Mbps / 50 ms   |                1.85574 |                  1.84495 |            -0.58% |              229.646 |                213.230 |       -7.15% |
| 5 Mbps / 100 ms  |                4.47391 |                  4.34868 |            -2.80% |              131.531 |                130.947 |       -0.44% |

### Results Summary

Across these six recorded scenarios:

* The hybrid controller achieved higher throughput in **2 out of 6 scenarios**.
* The hybrid controller reported lower mean delay in **all 6 scenarios**.
* Throughput was lower than Reno in four scenarios.
* The magnitude of the observed differences varied by network configuration.

These results indicate trade-offs between throughput and delay. They do not establish statistical significance or prove that the hybrid controller will perform better under all network conditions.

### Graphs and CSV Files

Check the `complete_project/results/` directory for available CSV outputs and generated graphs.

These files support the performance comparisons and provide more detail than the summary table alone.

---

## 📈 Understanding the Results

### Throughput

Throughput measures the rate at which data is successfully delivered over the measured interval. Higher throughput can indicate better link utilization, but it should be considered alongside delay and other metrics.

### Mean Delay

Mean delay represents the average delay reported for the measured flow. It is not necessarily equal to the configured link delay because network processing, queuing, and TCP behavior can affect the result.

### Congestion Window (CWND)

CWND controls how much unacknowledged data a TCP sender may have in flight. A smaller congestion window is not automatically better; it must be interpreted alongside throughput and delay.

### PPO Actions

The PPO policy selects among three actions:

| Action   | Intended behavior                                     |
| -------- | ----------------------------------------------------- |
| Decrease | Reduce the congestion window by a configured factor   |
| Maintain | Do not apply an additional PPO scaling change         |
| Increase | Increase the congestion window by a configured factor |

TCP window growth may still occur between controller decisions.

### Packet Loss

Packet-loss values are taken from the reported experiment statistics. FlowMonitor's `LostPackets` field should not automatically be treated as equal to the difference between transmitted and received packet counts.

---

## ⚠️ Limitations

* The PPO policy was evaluated in a synthetic environment and was not trained through direct interaction with the live ns-3 simulation.
* Some runtime controller measurements are proxies for the features used during model training.
* Runtime loss-event counters do not necessarily represent the same quantity as FlowMonitor packet-loss statistics.
* Only a limited number of network scenarios have been evaluated.
* The results do not establish statistical significance or universal superiority over TCP Reno.
* The website dashboard visualizes saved results and does not execute simulations itself.

These limitations should be considered when interpreting the experimental findings.

---

## 🚀 Future Enhancements

* Align runtime features with the model's training features.
* Train and evaluate the PPO policy against ns-3 or a validated equivalent environment.
* Evaluate more bandwidth, delay, and traffic configurations.
* Perform repeated experiments with different seeds and report confidence intervals.
* Improve logging to correlate controller decisions with network events.
* Add automated tests for model inference and feature preprocessing.
* Publish the website dashboard using GitHub Pages.
* Expand the dashboard with additional verified performance metrics.

---

## 👩‍💻 Author and Project Information

**Project:** Intelligent TCP Congestion Control Using Hybrid ML and Reinforcement Learning

**Repository:** [GitHub Project Repository](https://github.com/grandhiakshaya429-pixel/Intelligent-TCP-Congestion-Control)

This project is intended for academic experimentation and evaluation of ML-assisted congestion-control strategies.

---

## 📄 License

Add an appropriate license if the project is to be distributed or reused publicly. Until a license is specified, no additional reuse permissions are implied.
