import cv2
import torch
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

first_model = YOLO('best_detect.pt')
second_model = YOLO('best-classify.pt')

def is_valid_time(classification_results):
    
    if len(classification_results) != 4:
        return False
    hours, minutes = classification_results[:2], classification_results[2:]
    print(hours, minutes)
    try:
        hours = int(''.join(str(x) for x in hours))
        minutes = int(''.join(str(x) for x in minutes))
        if 0 <= hours < 24 and 0 <= minutes < 60:
            return f"{hours:02d}:{minutes:02d}"
    except ValueError:
        return False
    return False


def detect_and_classify(image_path):
    image = Image.open(image_path)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = first_model.predict(image_cv, imgsz=640)
    classification_results = []
    results = results[0]
    for i in range(len(results.boxes)):
        box = results.boxes[i]
        tensor = box.xyxy[0]
        x1 = int(tensor[0].item())
        y1 = int(tensor[1].item())
        x2 = int(tensor[2].item())
        y2 = int(tensor[3].item())
        crop_img = image_cv[y1:y2, x1:x2]
        #plt.imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        #plt.show()
        crop_results = []
        for i in range(4):
            width = crop_img.shape[1] // 4
            part_img = crop_img[:, i*width:(i+1)*width]
            #plt.imshow(cv2.cvtColor(part_img, cv2.COLOR_BGR2RGB))
            #plt.show()
            part_results = second_model.predict(part_img, imgsz=160)
            part_results = part_results[0]
            print("class:")
            class_name = part_results.probs.top1
            print(class_name)
            crop_results.append(class_name)

        classification_results.append(crop_results)

    return classification_results

def process_folder(folder_path, output_file):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name}...")
        results = detect_and_classify(file_path)
        print(results)
        time_representation = is_valid_time(results[0])
        with open(output_file, 'a')as file:
            if time_representation:
               file.write(f"{file_name}: {time_representation}\n")
            else:
               file.write(f"{file_name}: не время\n")


folder_path = 'test'
output_file = 'output'
process_folder(folder_path, output_file)
