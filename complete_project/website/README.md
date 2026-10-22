# Hybrid TCP Dashboard

A static, responsive project-demo website for the **Intelligent TCP Congestion Control Using Hybrid ML and Reinforcement Learning** project. It visualizes the saved comparison CSV and the included 10 Mbps / 50 ms controller traces. It does not launch ns-3 simulations.

## Run locally (recommended)

From the project root in Ubuntu/WSL:

```bash
cd website
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser. Keep the terminal open while presenting. Press `Ctrl+C` to stop the server.

You can also run the command from Windows PowerShell if Python is installed and you have navigated to the `website` directory:

```powershell
py -m http.server 8000
```

## Features

- Responsive dark-themed project dashboard
- Reno vs Hybrid ML-RL throughput and mean-delay charts
- Scenario selector and six-scenario comparison table
- CWND trace and PPO action counts from saved logs
- Architecture flow and project artifact links

Chart.js and Google Fonts are loaded from CDNs, so charts/fonts need internet access. The comparison table and other page structure are local. The charts use the supplied result files; they are not live simulation output.

## Data sources

- `../results/csv/hybrid_vs_reno_comparison.csv`
- `../results/csv/trial2_clean/hybrid_10Mbps_50ms_decisions.csv`
- `../results/csv/trial2_clean/hybrid_10Mbps_50ms_cwnd.csv`
- Existing project graphs in `../results/graphs/`

## Interpretation note

The dashboard preserves the recorded reported-loss values from the comparison CSV. Do not infer packet loss from TxPackets minus RxPackets. The project is an experimental prototype; see the root README and report for implementation details and limitations.
