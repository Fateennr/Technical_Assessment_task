from ultralytics import YOLO
import cv2

def detect_and_count(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]

    person_count = 0
    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if cls == 0:  # person
            person_count += 1
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        elif cls == 1:  # car
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)

    # Display count on image
    cv2.putText(img, f'People: {person_count}', (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    return img, person_count


model = YOLO('yolo26n.pt')

model.train(data='custom_dataset.yaml', epochs=50, imgsz=960, 
            batch=8, device=0, mosaic=1.0, close_mosaic=10, 
            scale=0.25, translate=0.08, degrees=3, fliplr=0.5, hsv_s=0.4, hsv_v=0.3)

# epoch ideally should be around 50-300, taking the lowest for now to test
# 960 or 1280 often performs noticeably better for pedestrians/cars in aerial imagery
# batch size 16 is stable


# check the dataset first
# train_image="/home/fateennr/Codes/Experiments/assessment_task/visDrone/VisDrone_Dataset/VisDrone2019-DET-train/images"
