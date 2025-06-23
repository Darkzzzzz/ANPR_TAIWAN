import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageFont, ImageDraw
from ultralytics import YOLO
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def text(img, text, xy=(0, 0), color=(0, 0, 0), size=20):
    pil = Image.fromarray(img)
    font = ImageFont.load_default()  # 使用默认字体
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color)
    return np.asarray(pil)

model = YOLO('./best.pt')

# 打开摄像机
cap = cv2.VideoCapture(0)

# 设置OCR字符白名单
custom_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

plate_coordinates = None  # 保存车牌坐标的变量
last_detection_time = 0  # 上次检测的时间戳

while True:
    ret, img = cap.read()
    if not ret:
        break

    current_time = time.time()
    if current_time - last_detection_time >= 3:  # 每隔3秒执行一次检测
        results = model.predict(img, save=False)
        boxes = results[0].boxes.xyxy

        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            tmp = cv2.cvtColor(img[y1:y2, x1:x2].copy(), cv2.COLOR_RGB2GRAY)
            license = pytesseract.image_to_string(tmp, lang='eng', config=custom_config)
            last_detection_time = current_time  # 更新上次检测的时间戳

            if license:
                # 保存车牌坐标并绘制车牌框
                plate_coordinates = (x1, y1, x2, y2)

    if plate_coordinates:
        x1, y1, x2, y2 = plate_coordinates
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = text(img, license, (x1, y1 - 20), (0, 255, 0), 25)

    cv2.imshow("yolov8_car", img)
    
    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像机并关闭窗口
cap.release()
cv2.destroyAllWindows()
