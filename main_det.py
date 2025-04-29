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
 title="Road Damage Detection API",
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

# Upload the model
model_det = YOLO("yolo11n.pt")

@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # async def detect_object_return_base64_img(file: bytes = File(...)):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Perform object detection with YOLOv11
    detections = model_det.predict(image, iou=0.8, conf=0.2, imgsz=512)
    for result in detections:
        results = result.save()  # display to screen

    return FileResponse(results, media_type="image/jpeg")
