from ultralytics import YOLO
import cv2
from PIL import Image

#model_path = '/home/skpark/git/pose-detection-keypoints-estimation-yolov8/runs/pose/train3/weights/last.pt'
model_path = '/home/skpark/git/pose-detection-keypoints-estimation-yolov8/last.pt'
image_path = '/home/skpark/git/pose-detection-keypoints-estimation-yolov8/data/images/train/bobcat_10007.jpg'
img = cv2.imread(image_path)

model = YOLO(model_path)

results = model(image_path)[0]
print(results)

 
for keypoint_indx, keypoint in enumerate(results[0].keypoints.xyn.cpu().numpy()[0]):
    print(keypoint_indx)
    print(keypoint)
    cv2.putText(img, str(keypoint_indx), (int(keypoint[0]), int(keypoint[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    
Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#cv2.imshow('img', img)
#exit
#cv2.waitKey(0)
