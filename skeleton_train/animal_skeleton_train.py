from ultralytics import YOLO
import os
model = YOLO('/home/skpark/git/pose-detection-keypoints-estimation-yolov8/last.pt')  # load a pretrained model (recommended for training)

#model = YOLO('/home/skpark/git/pose-detection-keypoints-estimation-yolov8/runs/pose/train6/weights/last.pt')  # load a pretrained model (recommended for training)
#model = YOLO('/home/skpark/git/pose-detection-keypoints-estimation-yolov8/yolov8n-pose.pt')
#model = YOLO('yolov8n-pose.pt')
model.train(data='config.yaml', epochs=10, imgsz=640 ,device=0)


#CUDA_VISIBLE_DEVICES=1,2,3,4 python train.py