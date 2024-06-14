import cv2
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient
import easyocr
#import pytesseract

#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="uU2L1z0e1EBGKvgwmm0J"
)

# Perform inference
image_path = 'images/test8.jpg'
result = CLIENT.infer(image_path, model_id="vehicle-registration-plates-trudk/2")
print(result)

# Load the image
image = cv2.imread(image_path)

# Extract the bounding box and snip the box out
for detection in result['predictions']:
    x, y, width, height, confidence = detection['x'], detection['y'], detection['width'], detection['height'], detection['confidence']
    if confidence > 0.80:
        top_left_x = int(x - width / 2)
        top_left_y = int(y - height / 2)
        bottom_right_x = int(x + width / 2)
        bottom_right_y = int(y + height / 2)
    
        # Crop the region of interest
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

#cropped_image = cv2.fastNlMeansDenoisingColored(cropped_image,None, 10, 10, 7, 15)
reader = easyocr.Reader(['en'])
output = reader.readtext(cropped_image)
print(output)
plate_num = ''
for words in output:
    word, conf = words[1], words[2]
    if conf > 0.35:
        plate_num = plate_num + word
        print(plate_num)

#result = pytesseract.image_to_string(cropped_image)
#print(result)

# Display the cropped image
plt.figure(figsize=(5, 5))
plt.title(str(plate_num))
plt.imshow(cropped_image)
plt.axis('off')
plt.show()