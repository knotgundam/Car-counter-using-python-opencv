import numpy as np
import cv2
import sys


video = cv2.VideoCapture(sys.argv[1])
if video == None :
	sys.exit("bug")
# BackgroundSubtractorMOG(history=None, nmixtures=None, backgroundRatio=None, noiseSigma=None)
fgbg = cv2.BackgroundSubtractorMOG(history=None, nmixtures=None, backgroundRatio=None, noiseSigma=None)
first = True
while(video.isOpened()):
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if first:
    	fgmask = fgbg.apply(frame)
    	first = False
    fgmask = fgbg.apply(frame,fgmask,0.1)
    fgmask = cv2.GaussianBlur(fgmask,(5,5),0)
    # cv2.imshow('frame',gray)
    cv2.imshow('frame',np.hstack( (gray, fgmask) ))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


