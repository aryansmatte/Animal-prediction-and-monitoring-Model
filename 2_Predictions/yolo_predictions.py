#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred():
    def __init__(self,onnx_model,data_yaml):
        # load YAML
        with open(data_yaml,mode='r') as f:
            data_yaml = yaml.load(f,Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']
        
        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    
        
    def predictions(self, image):
        orig_h, orig_w, d = image.shape
    
        # Step 1: Pad the image to a square without resizing
        max_rc = max(orig_h, orig_w)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[:orig_h, :orig_w] = image
    
        # Step 2: Resize the padded image for YOLO input
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()
    
        # Step 3: Process YOLO predictions
        detections = preds[0]
        boxes, confidences, classes = [], [], []
        x_factor, y_factor = max_rc / INPUT_WH_YOLO, max_rc / INPUT_WH_YOLO
    
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
    
                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
    
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    classes.append(class_id)
    
        # Step 4: Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45).flatten()
    
        # Step 5: Draw bounding boxes on the original image
        for i in indices:
            x, y, w, h = boxes[i]
            conf = int(confidences[i] * 100)
            label = f"{self.labels[classes[i]]}: {conf}%"
            color = self.generate_colors(classes[i])
    
            # Draw the bounding box on the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(image, (x, y - 20), (x + w, y), color, -1)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image

    
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
        return tuple(colors[ID])
        
        
    
    
    



