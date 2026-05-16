import cv2
import os
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_PATH  = '/home/fateennr/Codes/Experiments/assessment_task/detect-960px-mosaic-augmentation/train-8/weights/best.pt'
VIDEO_PATH  = '/home/fateennr/Codes/Experiments/assessment_task/tests/video2.mp4'
OUTPUT_DIR  = '/home/fateennr/Codes/Experiments/assessment_task/outputs'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'output_tracking2.mp4')

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# Track unique IDs seen so far
unique_person_ids = set()
unique_car_ids    = set()

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ── KEY DIFFERENCE: .track() instead of .predict() ────────────────
    results = model.track(
        frame,
        conf=0.25,
        device=0,
        tracker='bytetrack.yaml',
        persist=True,   
        verbose=False
    )[0]

    person_count = 0
    car_count    = 0

    if results.boxes.id is not None:   # IDs exist only when tracker assigns them
        for box, track_id in zip(results.boxes, results.boxes.id):
            cls      = int(box.cls)
            tid      = int(track_id)
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            if cls == 0:  # person
                person_count += 1
                unique_person_ids.add(tid)
                color = (0, 255, 0)
                label = f'P#{tid}'
            elif cls == 1:  # car
                car_count += 1
                unique_car_ids.add(tid)
                color = (0, 0, 255)
                label = f'C#{tid}'
            else:
                continue

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            cv2.putText(frame, label, (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── Overlay: current count & total unique ─────────────────────────
    cv2.rectangle(frame, (0,0), (280,90), (0,0,0), -1)
    cv2.putText(frame, f'People (frame):  {person_count}',  (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
    cv2.putText(frame, f'Cars   (frame):  {car_count}',     (10,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    cv2.putText(frame, f'Unique people:   {len(unique_person_ids)}', (10,74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,255,180), 2)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"  Frame {frame_idx}/{total_frames} | "
              f"People: {person_count} | Cars: {car_count} | "
              f"Unique people tracked: {len(unique_person_ids)}")

cap.release()
out.release()

print(f"\n✓ Saved: {OUTPUT_PATH}")
print(f"✓ Total unique people tracked: {len(unique_person_ids)}")
print(f"✓ Total unique cars tracked:   {len(unique_car_ids)}")