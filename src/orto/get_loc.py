import csv
import re
import matplotlib.pyplot as plt

def extract_sort_key(filename):
    match = re.search(r'(\d{4})\D*$', filename)
    return int(match.group(1)) if match else 0

def read_trajectory_from_csv(csv_path):
    trajectory = []

    with open(csv_path, "r") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader if row["latitude"] and row["longitude"]]

    data.sort(key=lambda row: extract_sort_key(row["filename"]))

    for row in data:
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        trajectory.append((lat, lon))

    return trajectory

def plot_trajectory(traj):
    lats, lons = zip(*traj)
    plt.figure(figsize=(8, 6))
    plt.plot(lons, lats, marker='o', linestyle='-', color='blue')
    plt.title("Trajektoria UAV z CSV")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

csv_file = "data/EPG_photos_gps.csv"
trajectory = read_trajectory_from_csv(csv_file)

print("📍 Trajektoria (posortowana):")
for i, (lat, lon) in enumerate(trajectory):
    print(f"{i+1:02d}: {lat:.6f}, {lon:.6f}")

plot_trajectory(trajectory)
