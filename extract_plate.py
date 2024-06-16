from ultralytics import YOLO
from PIL import Image
import os

license_dectector = YOLO('runs/detect/train10/weights/last.pt')
directory = "Cal_Cars/Car"
saveDirectory = "Cal_Cars/Plate"

if not os.path.exists(saveDirectory):
    os.makedirs(saveDirectory)

for file in os.listdir(directory):
    filename = os.path.join(directory, os.fsdecode(file))
    head, tail = os.path.split(filename)
    try:
        original_image = Image.open(filename)
        plate = license_dectector(original_image)
        for i, box in enumerate(plate[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cropped_image = original_image.crop((x1, y1, x2, y2))
            cropped_image.save(f"{saveDirectory}/{tail}")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
