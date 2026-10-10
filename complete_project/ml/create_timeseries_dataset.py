import csv
from pathlib import Path

BASE = Path("../results/csv")
OUT = Path("../ml/tcp_ml_dataset.csv")

scenarios = [
    ("10Mbps", "50ms", "Low"),
    ("10Mbps", "25ms", "Low"),
    ("10Mbps", "150ms", "Medium"),
    ("5Mbps", "100ms", "Medium"),
    ("3Mbps", "75ms", "Medium"),
    ("2Mbps", "50ms", "High"),
    ("1Mbps", "100ms", "High"),
]

all_rows = []

for bandwidth, delay, label in scenarios:

    stats_file = BASE / f"reno_{bandwidth}_{delay}_stats.csv"
    cwnd_file = BASE / f"reno_{bandwidth}_{delay}_cwnd.csv"
    flow_file = BASE / f"reno_{bandwidth}_{delay}_flow.csv"

    # Read FlowMonitor main data flow
    with open(flow_file, newline="") as f:
        flows = list(csv.DictReader(f))

    main_flow = next(row for row in flows if row["FlowId"] == "1")

    lost_packets = int(main_flow["LostPackets"])
    tx_packets = int(main_flow["TxPackets"])
    mean_delay = float(main_flow["MeanDelayMs"])

    loss_rate = (
        lost_packets / tx_packets * 100
        if tx_packets > 0 else 0
    )

    # Read CWND trace
    cwnd_rows = []

    with open(cwnd_file, newline="") as f:
        for row in csv.DictReader(f):
            cwnd_rows.append(row)

    # Read statistics
    stats_rows = []

    with open(stats_file, newline="") as f:
        for row in csv.DictReader(f):
            stats_rows.append(row)

    # Create one ML sample for every statistics timestamp
    for i, stat in enumerate(stats_rows):

        time = float(stat["Time"])
        total_rx = int(stat["TotalRxBytes"])

        # Find latest CWND measurement at or before this timestamp
        current_cwnd = 0

        for cwnd in cwnd_rows:
            if float(cwnd["Time"]) <= time:
                current_cwnd = int(cwnd["CwndNew"])
            else:
                break

        # Calculate approximate instantaneous throughput
        if i > 0:
            previous_rx = int(stats_rows[i - 1]["TotalRxBytes"])
            previous_time = float(stats_rows[i - 1]["Time"])

            dt = time - previous_time

            if dt > 0:
                throughput = (
                    (total_rx - previous_rx) * 8
                    / dt
                    / 1_000_000
                )
            else:
                throughput = 0
        else:
            throughput = 0

        all_rows.append({
            "time_s": time,
            "bandwidth_mbps": float(
                bandwidth.replace("Mbps", "")
            ),
            "configured_delay_ms": float(
                delay.replace("ms", "")
            ),
            "cwnd_bytes": current_cwnd,
            "total_rx_bytes": total_rx,
            "instant_throughput_mbps": throughput,
            "flow_mean_delay_ms": mean_delay,
            "flow_loss_rate_percent": loss_rate,
            "flow_lost_packets": lost_packets,
            "congestion_label": label
        })

OUT.parent.mkdir(parents=True, exist_ok=True)

fields = all_rows[0].keys()

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(all_rows)

print(f"Dataset created: {OUT}")
print(f"Total samples: {len(all_rows)}")

labels = {}

for row in all_rows:
    labels[row["congestion_label"]] = labels.get(
        row["congestion_label"], 0
    ) + 1

print("\nClass distribution:")

for label, count in labels.items():
    print(f"{label}: {count}")
