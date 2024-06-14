import cv2
def main():
    har = 'model/haarcascade_russian_plate_number.xml'

    img = cv2.imread('images/image1.jpg')

    plate_har = cv2.CascadeClassifier(har)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plate = plate_har.detectMultiScale(img_gray, 1.1, 4)

    for(x, y, w, h) in plate:
        area = w * h

        if(area > 500):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_temp = img[y: y + h, x: x + w]
            cv2.imwrite("iamges/test1_after.jpg", img_temp)
    

if __name__ == "__main__":
    main()