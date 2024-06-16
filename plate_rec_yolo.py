from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np
import matplotlib.pyplot as plt

#model = YOLO('yolov10n.pt')
#img = model('images/test1.jpg')
#img[0].show()

license_dectector = YOLO('runs/detect/train10/weights/last.pt')
original_image = Image.open('images/test8.jpg')

plate = license_dectector(original_image)
plate[0].show()
print(plate)


# Iterate over detected bounding boxes and crop the image
for i, box in enumerate(plate[0].boxes):
    x1, y1, x2, y2 = box.xyxy[0]  
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_image = original_image.crop((x1, y1, x2, y2))
    

cropped_image_np = np.array(cropped_image)
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image_np)
print(result)

# Display the cropped image
plt.figure(figsize=(5, 5))
plt.title(str(result[0][1]))
plt.imshow(cropped_image)
plt.axis('off')
plt.show()