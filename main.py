from fastapi import FastAPI
from fastapi import File, UploadFile

import cv2
import numpy as np
from ultralytics import YOLO

from starlette.responses import Response
from fastapi.responses import FileResponse
import yaml
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
 title="Invij - Crack Detection & Classification API",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/notify/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')

# model_cls = YOLO("yolo11m-cls.pt")
# model_det = YOLO("crackm-det-best-3.pt")
model_det = YOLO("bestdet.pt")
model_cls = YOLO("crackm-cls-best.pt")

with open('cls_name_pupr.yml', 'r') as file:
    cls_name = yaml.safe_load(file)

# model = YOLO("yolov8n.pt")
@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # async def detect_object_return_base64_img(file: bytes = File(...)):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    # image_bytes = get_image_from_bytes(file)
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv11
    detections = model_det.predict(image, iou=0.8, conf=0.2, imgsz=512)
    # print(type(detections))
    # print(detections)
    for result in detections:
        box = result.boxes  # Boxes object for bounding box outputs
        mask = result.masks  # Masks object for segmentation masks outputs
        keypoint = result.keypoints  # Keypoints object for pose outputs
        prob = result.probs  # Probs object for classification outputs
        ob = result.obb  # Oriented boxes object for OBB outputs
        # results = result.save()  # display to screen

    bbox = box.data
    bbox = bbox.numpy()

    # label = prob
    # print(cls_name['names'][label.top1])

    for det in bbox:
    #    print(det[:4])
       x1, y1, x2, y2 = map(int, det[:4])
       crop_img = image[y1:y2, x1:x2]
       classification = model_cls(crop_img)
       for cls in classification:
          label = cls.probs
       label = cls_name['names'][label.top1]
       print(label)

       if label == "tidak_parah": 
           cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Draw bounding box
           cv2.putText(image, label, (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Add label
       else:
           cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Draw bounding box
           cv2.putText(image, label, (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Add label
       

    cv2.imwrite('combined_result.jpg', image)

    return FileResponse('combined_result.jpg', media_type="image/jpeg")
