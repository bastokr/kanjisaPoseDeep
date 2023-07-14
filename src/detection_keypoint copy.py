import sys
import cv2
import numpy as np
from pydantic import BaseModel

import ultralytics
from ultralytics.yolo.engine.results import Results

# Define keypoint
class GetKeypoint(BaseModel):
   
    
    NOSE:           int = 0
    UPPER_JAW:           int = 1
    LOWER_JAW:           int = 2
    MOUTH_END_RIGHT:           int = 3
    MOUTH_END_LEFT:           int = 4
    RIGHT_EYE:           int = 5
    RIGHT_EARBASE:           int = 6
    RIGHT_EAREND:           int = 7
    RIGHT_ANTLER_BASE:           int = 8
    RIGHT_ANTLER_END:           int = 9
    LEFT_EYE:           int = 10
    LEFT_EARBASE:           int = 11
    LEFT_EAREND:           int = 12
    LEFT_ANTLER_BASE:           int = 13
    LEFT_ANTLER_END:           int = 14
    NECK_BASE:           int = 15
    NECK_END:           int = 16
    THROAT_BASE:           int = 17
    THROAT_END:           int = 18
    BACK_BASE:           int =19
    BACK_END:           int = 20
    BACK_MIDDLE:           int = 21
    TAIL_BASE:           int = 22
    TAIL_END:           int = 23
    FRONT_LEFT_THAI:           int = 24
    FRONT_LEFT_KNEE:           int = 25
    FRONT_LEFT_PAW:           int = 26
    FRONT_RIGHT_THAI:           int = 27
    FRONT_RIGHT_PAW:           int = 28
    FRONT_RIGHT_KNEE:           int = 29
    BACK_LEFT_KNEE:           int = 30
    BACK_LEFT_PAW:           int = 31
    BACK_LEFT_THAI:           int = 32
    BACK_RIGHT_THAI:           int = 33
    BACK_RIGHT_PAW:           int = 34
    BACK_RIGHT_KNEE:           int = 35
    BELLY_BOTTOM:           int = 36
    BODY_MIDDLE_RIGHT:           int = 37
    BODY_MIDDLE_LEFT:           int = 39


class DetectKeypoint:
    def __init__(self, yolov8_model='yolov8m-pose'):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()
 
    def __load_model(self):
        if not self.yolov8_model.split('-')[-1] == 'pose':
            sys.exit('Model not yolov8 pose')
        self.model = ultralytics.YOLO(model=self.yolov8_model)

        # extract function keypoint
    def extract_keypoint(self, keypoint: np.ndarray) -> list:
        # nose
        nose_x, nose_y = keypoint[self.get_keypoint.NOSE]
        upper_jaw_x,upper_jaw_y  = keypoint[self.get_keypoint.UPPER_JAW]
        mouth_end_right_x,mouth_end_right_y = keypoint[self.get_keypoint.MOUTH_END_RIGHT]
        mouth_end_left_x,mouth_end_left_y = keypoint[self.get_keypoint.MOUTH_END_LEFT]
        right_eye_x,right_eye_y  = keypoint[self.get_keypoint.RIGHT_EYE]
        right_earbase_x,right_earbase_y = keypoint[self.get_keypoint.RIGHT_EARBASE]
        right_earend_x,right_earend_y = keypoint[self.get_keypoint.RIGHT_EAREND]
        right_antler_base_x,right_antler_base_y = keypoint[self.get_keypoint.RIGHT_ANTLER_BASE]
        right_antler_end_x,right_antler_end_y = keypoint[self.get_keypoint.RIGHT_ANTLER_END]
        left_eye_x,left_eye_y= keypoint[self.get_keypoint.LEFT_EYE]
        left_earbase_x,left_earbase_y = keypoint[self.get_keypoint.LEFT_EARBASE]
        left_earend_x,left_earend_y = keypoint[self.get_keypoint.LEFT_EAREND]
        left_antler_base_x,left_antler_base_y = keypoint[self.get_keypoint.LEFT_ANTLER_BASE]
        left_antler_end_x,left_antler_end_y = keypoint[self.get_keypoint.LEFT_ANTLER_END]
        neck_base_x,neck_base_y = keypoint[self.get_keypoint.NECK_BASE]
        neck_end_x,neck_end_y = keypoint[self.get_keypoint.NECK_END]
        throat_base_x,throat_base_y = keypoint[self.get_keypoint.THROAT_BASE]
        throat_end_x,throat_end_y = keypoint[self.get_keypoint.THROAT_END]
        back_base_x,back_base_y = keypoint[self.get_keypoint.BACK_BASE]
        back_end_x,back_end_y = keypoint[self.get_keypoint.BACK_END]
        back_middle_x,back_middle_y = keypoint[self.get_keypoint.BACK_MIDDLE]
        tail_base_x,tail_base_y = keypoint[self.get_keypoint.TAIL_BASE]
        tail_end_x,tail_end_y = keypoint[self.get_keypoint.TAIL_END]
        front_left_thai_x,front_left_thai_y = keypoint[self.get_keypoint.FRONT_LEFT_THAI]
        front_left_knee_x,front_left_knee_y = keypoint[self.get_keypoint.FRONT_LEFT_KNEE]
        front_left_paw_x,front_left_paw_y = keypoint[self.get_keypoint.FRONT_LEFT_PAW]
        front_right_thai_x,front_right_thai_y = keypoint[self.get_keypoint.FRONT_RIGHT_THAI]
        front_right_paw_x,front_right_paw_y = keypoint[self.get_keypoint.FRONT_RIGHT_PAW]
        front_right_knee_x,front_right_knee_y = keypoint[self.get_keypoint.FRONT_RIGHT_KNEE]
        back_left_knee_x,back_left_knee_y = keypoint[self.get_keypoint.BACK_LEFT_KNEE]
        back_left_paw_x,back_left_paw_y = keypoint[self.get_keypoint.BACK_LEFT_PAW]
        back_left_thai_x,back_left_thai_y = keypoint[self.get_keypoint.BACK_LEFT_THAI]
        back_right_thai_x,back_right_thai_y = keypoint[self.get_keypoint.BACK_RIGHT_THAI]
        back_right_paw_x,back_right_paw_y = keypoint[self.get_keypoint.BACK_RIGHT_PAW]
        back_right_knee_x,back_right_knee_y = keypoint[self.get_keypoint.BACK_RIGHT_KNEE]
        belly_bottom_x,belly_bottom_y = keypoint[self.get_keypoint.BELLY_BOTTOM]
        body_middle_right_x,body_middle_right_y = keypoint[self.get_keypoint.BODY_MIDDLE_RIGHT]
        body_middle_left_x,body_middle_left_y = keypoint[self.get_keypoint.BODY_MIDDLE_LEFT]

        
        return [
            nose_x, nose_y ,upper_jaw_x,upper_jaw_y  ,mouth_end_right_x,mouth_end_right_y ,mouth_end_left_x,mouth_end_left_y ,right_eye_x,right_eye_y  ,
        right_earbase_x,right_earbase_y,right_earend_x,right_earend_y ,right_antler_base_x,right_antler_base_y ,
        right_antler_end_x,right_antler_end_y,left_eye_x,left_eye_y,
        left_earbase_x,left_earbase_y, left_earend_x,left_earend_y ,
        left_antler_base_x,left_antler_base_y,left_antler_end_x,left_antler_end_y ,
        neck_base_x,neck_base_y , neck_end_x,neck_end_y ,
        throat_base_x,throat_base_y,throat_end_x,throat_end_y ,
        back_base_x,back_base_y ,back_end_x,back_end_y ,
        back_middle_x,back_middle_y, tail_base_x,tail_base_y,
        tail_end_x,tail_end_y , front_left_thai_x,front_left_thai_y,
        front_left_knee_x,front_left_knee_y ,front_left_paw_x,front_left_paw_y ,
        front_right_thai_x,front_right_thai_y ,front_right_paw_x,front_right_paw_y ,
        front_right_knee_x,front_right_knee_y ,    back_left_knee_x,back_left_knee_y ,
        back_left_paw_x,back_left_paw_y , back_left_thai_x,back_left_thai_y ,
        back_right_thai_x,back_right_thai_y,  back_right_paw_x,back_right_paw_y ,
        back_right_knee_x,back_right_knee_y,  belly_bottom_x,belly_bottom_y ,
        body_middle_right_x,body_middle_right_y ,  body_middle_left_x,body_middle_left_y   ]
    
    def get_xy_keypoint(self, results: Results) -> list:
        result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data
    
    def __call__(self, image: np.array) -> Results:
        results = self.model.predict(image, save=False)[0]
        return results

