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
    │   ├── conges
```
