from ultralytics import YOLO
from PIL import Image
from glob import glob
import pandas as pd
import cv2 as cv
import time


model = YOLO("runs/detect/train3/weights/best.pt")
# cap = cv.VideoCapture("/home/jimmy/Downloads/Dash Cam -  driving in Poland _ вождение в Польше _ Fahren in Polen-7Wdrr_VnLpU.mkv")
cap = cv.VideoCapture("/home/jimmy/Downloads/Daytime driving 30 fps forward facing road footage 05-05-2017 Journey 2-rGgAxnHq2vs.mkv")
prev = 0
while True:
    prev = time.time()
    # ret, frame = cap.read() # skip a frame
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, verbose=False)
    for r in results:
        if len(r.boxes.conf) and r.boxes.conf.max() > 0.75:
            print(r.boxes)
            prediction = int(r.boxes.cls[r.boxes.conf.argmax()])
            print(r.names[prediction])
    cv.imshow('frame', cv.resize(frame, (960, 540)) )
    key  = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('d'):
        cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) + 60)
    if key == ord('a'):
        cap.set(cv.CAP_PROP_POS_FRAMES, cap.get(cv.CAP_PROP_POS_FRAMES) - 60)
    now = time.time()
    waitfor = (1/30)-(now-prev)
    if waitfor > 0:
        time.sleep(waitfor)

# false_positivess = [0] * 9
# false_negativess = [0] * 9
# true_positivess = [0] * 9
# true_negativess = [0] * 9
# threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# inference_time = 0
# for file in glob("GTSDB/test/*.jpg"):
#     expected = pd.read_csv(file[:-4] + ".txt", delimiter=' ', names=['ClassID', 'x_center', 'y_center', 'width', 'height'])
#     expected = expected[expected['ClassID'] <= 8]
#     im = Image.open(file)

#     results = model.predict(source=im, save=False)

#     for i, thresh in enumerate(threshs):
#         for r in results:
#             inference_time += r.speed['inference']
#             if len(r.boxes.conf) and r.boxes.conf.max() > thresh:
#                 prediction = int(r.boxes.cls[r.boxes.conf.argmax()])
#                 if (expected['ClassID'] == prediction).any():
#                     true_positivess[i] += 1
#                 else:
#                     false_positivess[i] += 1
#             else:
#                 if bool(len(expected)):
#                     false_negativess[i] += 1
#                 else:
#                     true_negativess[i] += 1
# print(f"average inference_time (ms): {inference_time / len(glob("GTSDB/test/*.jpg"))}")
        
# import matplotlib.pyplot as plt

# plt.plot(threshs, false_positivess, label='false positives')
# plt.plot(threshs, false_negativess, label='false negatives')
# plt.plot(threshs, true_positivess, label='true positives')
# plt.plot(threshs, true_negativess, label='true negatives')
# plt.xlabel("confidence threshold")
# plt.legend(loc='upper right')
# plt.savefig("confidence")