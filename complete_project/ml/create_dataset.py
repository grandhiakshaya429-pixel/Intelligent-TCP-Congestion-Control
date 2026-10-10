import csv
from pathlib import Path

BASE = Path("../results/csv")
OUT = Path("../ml/tcp_congestion_dataset.csv")

scenarios = [
    ("10Mbps", "50ms", "Low"),
    ("10Mbps", "25ms", "Low"),
    ("10Mbps", "150ms", "Medium"),
    ("5Mbps", "100ms", "Medium"),
    ("3Mbps", "75ms", "Medium"),
    ("2Mbps", "50ms", "High"),
    ("1Mbps", "100ms", "High"),
]

rows = []

for bandwidth, delay, label in scenarios:
    filename = BASE / f"reno_{bandwidth}_{delay}_flow.csv"

    with open(filename, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row["FlowId"] != "1":
                continue

            tx_packets = int(row["TxPackets"])
            rx_packets = int(row["RxPackets"])
            lost_packets = int(row["LostPackets"])
            throughput = float(row["ThroughputMbps"])
            mean_delay = float(row["MeanDelayMs"])

            loss_rate = (
                lost_packets / tx_packets * 100
                if tx_packets > 0 else 0
            )

            rows.append({
                "bandwidth_mbps": float(bandwidth.replace("Mbps", "")),
                "delay_ms": float(delay.replace("ms", "")),
                "throughput_mbps": throughput,
                "mean_delay_ms": mean_delay,
                "tx_packets": tx_packets,
                "rx_packets": rx_packets,
                "lost_packets": lost_packets,
                "loss_rate_percent": loss_rate,
                "congestion_label": label
            })

OUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUT, "w", newline="") as f:
    fieldnames = rows[0].keys()
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Dataset created: {OUT}")
print(f"Rows: {len(rows)}")

for row in rows:
    print(
        row["bandwidth_mbps"],
        "Mbps |",
        row["delay_ms"],
        "ms |",
        row["throughput_mbps"],
        "Mbps |",
        row["mean_delay_ms"],
        "ms |",
        row["congestion_label"]
    )
