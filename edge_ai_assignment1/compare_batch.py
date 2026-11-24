
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

# Create 400 valid test readings
rows = [
    {
        "battery_v": 3.7 + (i % 10) * 0.01,
        "temp_c": 25 + (i % 15),
        "rings_per_day": 20 + (i % 100),
        "wifi_rssi_dbm": -60 - (i % 30),
        "age_months": 6 + (i % 24),
    }
    for i in range(400)
]

# Test 1: 400 individual requests
single_predictions = []
start = time.perf_counter()

for row in rows:
    response = requests.post(f"{BASE_URL}/predict", json=row)
    response.raise_for_status()
    single_predictions.append(response.json()["days_left"])

single_time = time.perf_counter() - start

# Test 2: One batch request containing 400 readings
start = time.perf_counter()

response = requests.post(
    f"{BASE_URL}/predict_batch",
    json={"rows": rows},
)
response.raise_for_status()

batch_time = time.perf_counter() - start
batch_predictions = response.json()["days_left"]

# Compare results
differences = [
    abs(a - b)
    for a, b in zip(single_predictions, batch_predictions)
]

print("400 individual requests:", round(single_time * 1000, 2), "ms")
print("One batch of 400:", round(batch_time * 1000, 2), "ms")
print("Individual predictions:", len(single_predictions))
print("Batch predictions:", len(batch_predictions))
print("Maximum prediction difference:", max(differences))
print("Number of differences above 0.001:", sum(d > 0.001 for d in differences))
print("First 5 individual predictions:", single_predictions[:5])
print("First 5 batch predictions:", batch_predictions[:5])
print("Predictions match within 0.001:", max(differences) < 0.001)
print("Speed improvement:", round(single_time / batch_time, 2), "times")
