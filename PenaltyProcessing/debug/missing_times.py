from datetime import datetime
with open(
    "/home/josh/BA/Download_CSV/games_position_files/Frisch_Auf!_Goeppingen_vs_Rhein_Neckar_Loewen_2_phases_positions.csv",
    "r",
) as f:
    relevant_times = []
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue  # skip header
        values = line.strip().split(";")
        time = values[1]
        minute = time.split(",")[1].strip().split(":")[1]
        fmt = "%m/%d/%Y, %H:%M:%S.%f %p"
        if int(minute) > 37:
            relevant_times.append(time)
print("Relevant times:")
for t in relevant_times:
    print(t)
