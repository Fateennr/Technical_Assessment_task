import os

input_dir = '/home/fateennr/Codes/Experiments/assessment_task/visDrone/VisDrone_Dataset/VisDrone2019-DET-train/labels'
output_dir = '/home/fateennr/Codes/Experiments/assessment_task/visDrone/VisDrone_Dataset/VisDrone2019-DET-train/labels_filtered'

os.makedirs(output_dir, exist_ok=True)

# remapping to new IDs
CLASS_MAP = {
    0: 0,  # pedestrian and people → person, car
    1: 0,
    3: 1, 
}

for filename in os.listdir(input_dir):
    if not filename.endswith('.txt'):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, 'r') as f:
        lines = f.readlines()

    filtered = []
    for line in lines:
        print(line)
        parts = line.strip().split()
        if not parts:
            continue

        cls = int(parts[0])
        if cls in CLASS_MAP:
            new_cls = CLASS_MAP[cls]
            new_line = f"{new_cls} {' '.join(parts[1:])}\n"
            filtered.append(new_line)

    with open(output_path, 'w') as f:
        f.writelines(filtered)

print("Done!")