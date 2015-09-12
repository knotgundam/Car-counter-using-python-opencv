import numpy as np
import cv2


video = cv2.VideoCapture('vdo/Export Folder(1)/fix01.avi')
# BackgroundSubtractorMOG(history=None, nmixtures=None, backgroundRatio=None, noiseSigma=None)
fgbg = cv2.BackgroundSubtractorMOG()

while(video.isOpened()):
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(frame)
    # cv2.imshow('frame',gray)
    cv2.imshow('frame',fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


