
import numpy as np

def read_tle_file(file_path):
    """Reads a TLE file and returns a list of dictionaries with orbital elements."""
    sats = []
    with open(file_path) as f:
        lines = f.readlines()
    
    l1 = ""
    l2 = ""
    name = ""
    for line in lines:
        line = line.strip()
        if line.startswith('1 '):
            l1 = line
        elif line.startswith('2 '):
            l2 = line
            name = l2.split()[1]
            inclination = float(l2.split()[2])
            raan = float(l2.split()[3])
            sats.append({"name": name, "l1": l1, "l2": l2, "inc": inclination, "raan": raan})
    return sats

def find_best_pairs(sats, num_pairs=5, window_size=100):
    """Finds the pairs of satellites with the most similar orbital planes using a windowed search."""
    
    # Sort satellites by inclination
    sats.sort(key=lambda s: s["inc"])

    pairs = []
    for i in range(len(sats)):
        # Define the window of satellites to compare against
        start_index = i + 1
        end_index = min(i + 1 + window_size, len(sats))
        
        for j in range(start_index, end_index):
            inc_diff = abs(sats[i]["inc"] - sats[j]["inc"])
            raan_diff = abs(sats[i]["raan"] - sats[j]["raan"])
            if raan_diff > 180:
                raan_diff = 360 - raan_diff
            diff = inc_diff + 0.5 * raan_diff
            pairs.append(((sats[i], sats[j]), diff))

    # Sort all found pairs by their difference metric and return the best ones
    pairs.sort(key=lambda x: x[1])
    return [p[0] for p in pairs[:num_pairs]]

def main():
    tle_file = 'leo_satellites.txt'
    sats = read_tle_file(tle_file)
    
    if len(sats) < 2:
        print("Error: Not enough satellites in the TLE file.")
        return

    best_pairs = find_best_pairs(sats)

    print("Found best pairs of satellites for transfer:")
    for i, (sat1, sat2) in enumerate(best_pairs):
        print(f"\nPair {i+1}:")
        print(f"  Satellite 1: {sat1['name']} (Inc: {sat1['inc']}, RAAN: {sat1['raan']})")
        print(f"  Satellite 2: {sat2['name']} (Inc: {sat2['inc']}, RAAN: {sat2['raan']})")

if __name__ == '__main__':
    main()
