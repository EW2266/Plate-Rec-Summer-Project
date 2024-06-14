import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import time
import os
import datetime
import uuid

def main():
    #print('Running')

    '''
    cap = cv2.VideoCapture(0)
    time.sleep(3)
    sucess, img = cap.read()
    plt.imshow(img)

    now = str(datetime.datetime.now())
    imgname = os.path.join('images' + '/'+'{}.jpg'.format(str(uuid.uuid1())))
    print(imgname)
    cv2.imwrite(imgname, img)
    '''

    img = cv2.imread('images/test2.jpg')
    imgname = os.path.join('images' + '/'+'{}.jpg'.format(str(uuid.uuid1())))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b_filter = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(b_filter, 30, 200)

    #cv2.imwrite(imgname, cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))

    keypoints = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 20, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [location], 0,255, -1)

    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]

    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    if not result:
        raise ValueError("No text detected by OCR.")
    else:
        print(result)
    
    text = result[0][-2]
    print(text)

    return text

if __name__ == "__main__":
    main()