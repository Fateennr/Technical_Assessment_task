import cv2
import os
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_PATH  = '/home/fateennr/Codes/Experiments/assessment_task/detect-960px-mosaic-augmentation/train-8/weights/best.pt'
VIDEO_PATH  = '/home/fateennr/Codes/Experiments/assessment_task/tests/video1.mp4'
OUTPUT_DIR  = '/home/fateennr/Codes/Experiments/assessment_task/outputs'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'output_video.mp4')

# ── Pre-flight checks ──────────────────────────────────────────────────
print("=" * 50)
print("Checking paths...")
print("=" * 50)

errors = []

if not os.path.exists(MODEL_PATH):
    errors.append(f"  ✗ Model not found:  {MODEL_PATH}")
else:
    print(f"  ✓ Model found:      {MODEL_PATH}")

if not os.path.exists(VIDEO_PATH):
    errors.append(f"  ✗ Video not found:  {VIDEO_PATH}")
else:
    print(f"  ✓ Video found:      {VIDEO_PATH}")

if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"  ✓ Output dir created: {OUTPUT_DIR}")
    except Exception as e:
        errors.append(f"  ✗ Cannot create output dir: {e}")
else:
    print(f"  ✓ Output dir exists:  {OUTPUT_DIR}")

if errors:
    print("\n" + "=" * 50)
    print("ERRORS — fix these before running:")
    for e in errors:
        print(e)
    print("=" * 50)
    exit(1)

print("=" * 50)
print("All checks passed. Starting...\n")

# ── Load model ─────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)

# ── Video setup ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: OpenCV could not open video: {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

if not out.isOpened():
    print(f"ERROR: VideoWriter could not open output path: {OUTPUT_PATH}")
    cap.release()
    exit(1)

# ── Process frames ─────────────────────────────────────────────────────
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, device=0, verbose=False)[0]
    person_count = 0
    car_count    = 0

    for box in results.boxes:
        cls = int(box.cls)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:
            person_count += 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
        elif cls == 1:
            car_count += 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)

    cv2.rectangle(frame, (0,0), (220,65), (0,0,0), -1)
    cv2.putText(frame, f'People: {person_count}', (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(frame, f'Cars  : {car_count}',   (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 50 == 0:
        print(f"  Processed {frame_idx}/{total_frames} frames...")

cap.release()
out.release()

# ── Post-flight check ──────────────────────────────────────────────────
print()
if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"✓ Output saved:  {OUTPUT_PATH}")
    print(f"✓ File size:     {size_mb:.2f} MB")
    print(f"✓ Frames written: {frame_idx}")
else:
    print(f"✗ Output file missing or empty: {OUTPUT_PATH}")
    print("  Something went wrong with VideoWriter.")