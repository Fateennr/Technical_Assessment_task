import cv2
import os
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_PATH  = '/home/fateennr/Codes/Experiments/assessment_task/detect-960px-mosaic-augmentation/train-8/weights/best.pt'
VIDEO_PATH  = '/home/fateennr/Codes/Experiments/assessment_task/tests/video2.mp4'
OUTPUT_DIR  = '/home/fateennr/Codes/Experiments/assessment_task/outputs'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'output_deepsort.mp4')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model & tracker ───────────────────────────────────────────────
model   = YOLO(MODEL_PATH)
tracker = DeepSort(
    max_age=30,          # frames to keep a lost track alive
    n_init=3,            # detections needed before confirming a track
    max_cosine_distance=0.3,
    nn_budget=100
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps          = cap.get(cv2.CAP_PROP_FPS)
w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

unique_person_ids = set()
unique_car_ids    = set()

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ── Step 1: YOLO detection ─────────────────────────────────────────
    results = model(frame, conf=0.25, device=0, verbose=False)[0]

    # ── Step 2: Convert detections to DeepSORT format ─────────────────
    # DeepSORT expects: [ ([x, y, w, h], confidence, class_id), ... ]
    detections = []
    for box in results.boxes:
        cls  = int(box.cls)
        if cls not in [0, 1]:      # only person and car
            continue
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw = x2 - x1
        bh = y2 - y1
        detections.append(([x1, y1, bw, bh], conf, cls))

    # ── Step 3: Update tracker ─────────────────────────────────────────
    tracks = tracker.update_tracks(detections, frame=frame)

    person_count = 0
    car_count    = 0

    for track in tracks:
        if not track.is_confirmed():   # skip unconfirmed tracks
            continue

        tid = track.track_id
        cls = track.det_class
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # left,top,right,bottom

        if cls == 0:   # person
            person_count += 1
            unique_person_ids.add(tid)
            color = (0, 255, 0)
            label = f'Person #{tid}'
        elif cls == 1: # car
            car_count += 1
            unique_car_ids.add(tid)
            color = (0, 0, 255)
            label = f'Car #{tid}'
        else:
            continue

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
        cv2.putText(frame, label, (x1, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ── Overlay ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (280,90), (0,0,0), -1)
    cv2.putText(frame, f'People (frame):  {person_count}',       (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)
    cv2.putText(frame, f'Cars   (frame):  {car_count}',          (10,48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)
    cv2.putText(frame, f'Unique people:   {len(unique_person_ids)}', (10,74),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,255,180), 2)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"  Frame {frame_idx}/{total_frames} | "
              f"People: {person_count} | Cars: {car_count} | "
              f"Unique: {len(unique_person_ids)}")

cap.release()
out.release()

print(f"\n✓ Saved: {OUTPUT_PATH}")
print(f"✓ Unique people tracked: {len(unique_person_ids)}")
print(f"✓ Unique cars tracked:   {len(unique_car_ids)}")