import csv
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA = Path("tcp_ml_dataset.csv")
MODEL = Path("congestion_model.pkl")

FEATURES = [
    "bandwidth_mbps",
    "configured_delay_ms",
    "cwnd_bytes",
    "total_rx_bytes",
    "instant_throughput_mbps",
    "flow_mean_delay_ms",
    "flow_loss_rate_percent",
    "flow_lost_packets",
]

rows = []

with open(DATA, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)


def make_features(row):
    return [
        float(row[name])
        for name in FEATURES
    ]


scenarios = sorted(
    {
        (
            float(row["bandwidth_mbps"]),
            float(row["configured_delay_ms"])
        )
        for row in rows
    }
)

print("\n===== LEAVE-ONE-SCENARIO-OUT RANDOM FOREST =====")
print(f"Total samples: {len(rows)}")
print(f"Total scenarios: {len(scenarios)}")

results = []

for test_scenario in scenarios:

    train_rows = [
        row for row in rows
        if (
            float(row["bandwidth_mbps"]),
            float(row["configured_delay_ms"])
        ) != test_scenario
    ]

    test_rows = [
        row for row in rows
        if (
            float(row["bandwidth_mbps"]),
            float(row["configured_delay_ms"])
        ) == test_scenario
    ]

    X_train = [make_features(row) for row in train_rows]
    y_train = [row["congestion_label"] for row in train_rows]

    X_test = [make_features(row) for row in test_rows]
    y_test = [row["congestion_label"] for row in test_rows]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results.append(accuracy)

    print(
        f"\nTest scenario: "
        f"{test_scenario[0]:g} Mbps / "
        f"{test_scenario[1]:g} ms"
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


average_accuracy = sum(results) / len(results)

print("\n===== SUMMARY =====")
print(f"Average leave-one-scenario-out accuracy: {average_accuracy:.4f}")


# Train final model on the complete dataset.
X_all = [make_features(row) for row in rows]
y_all = [row["congestion_label"] for row in rows]

final_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

final_model.fit(X_all, y_all)

with open(MODEL, "wb") as f:
    pickle.dump(final_model, f)

print(f"\nFinal model trained on all {len(rows)} samples.")
print(f"Model saved to: {MODEL}")
