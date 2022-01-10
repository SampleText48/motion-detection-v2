from skimage.metrics import structural_similarity
import numpy as np
import cv2 as cv
import imutils
import time
import argparse

		##argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=100, help="minimum area size")
args = vars(ap.parse_args())

	##captures the webcam (webcam 1 is the built-in one, 0 is the usb)
#cap = cv.VideoCapture('added bugs no light.mp4')
#cap = cv.VideoCapture('newCamTest2.avi')
cap = cv.VideoCapture(0)
if not cap.isOpened():
	print('can not open')
	exit()
firstFrame = None
lastDelta = None
##contourFrame = None
##startTime = round(time.time() * 1000)

	##continuously reads frames from the cam
while True:
	text = 'not detected'
	ret, frame = cap.read()
	if not ret:
		print('can not read frame')
		break
			##makes frame grayscale
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	#cv.imshow('frame', gray)

			##exits program if q is pressed
	if cv.waitKey(1) == ord('q'):
		break
	
	if firstFrame is None:
		firstFrame = gray
		continue

			##gets frame difference between current frame and previous frame
	frameDelta = cv.absdiff(firstFrame, gray)

			##dialates thresh
	thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]
	thresh = cv.dilate(thresh, None, iterations=2)
	cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

			##draws contours
	for c in cnts:
		if cv.contourArea(c) < args["min_area"]:
			continue
		(x, y, w, h) = cv.boundingRect(c)
		cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "DETECTED"


	if lastDelta is None:
		lastDelta = frameDelta
		continue
	
			##just testing what filters do
	##kernel = np.ones((5,5), np.float32)/25
	##dst = cv.filter2D(frameDelta, -1, kernel)
	##blur = cv.blur(frameDelta, (5, 5))
	##medianbl = cv.medianBlur(frameDelta, 5)
	##gaussbl = cv.GaussianBlur(frameDelta, (5, 5), 0)
	bilatFilter = cv.bilateralFilter(frameDelta, 9, 75, 75)
	bilatFilterLast = cv.bilateralFilter(lastDelta, 9, 75, 75)

			##blob detection using sift
	sift = cv.SIFT_create()
	kp = sift.detect(bilatFilter, None)
	##kp = sift.detect(frameDelta, None)
	blobs = bilatFilter
	##blobs = frameDelta
	blobs = cv.drawKeypoints(bilatFilter, kp, blobs)
	##blobs = cv.drawKeypoints(frameDelta, kp, blobs)

			##blob detection using surf [NOT WORKING] surf algorithm patented
	##surf = cv.xfeatures2d.SURF_create(50000)
	##surf.setHessianThreshold(50000)
	##kp, des = surf.detectAndCompute(bilatFilter, None)
	##blobs = cv.drawKeypoints(bilatFilter, kp, None, (255, 0, 0), 4)
	
			##shows mesage if motion detected 
	cv.putText(frame, "Motion: {}".format(text), (10, 20),
	cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
	##cv.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		##(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	
			##video output
	##cv.imshow('bilateral filtering', bilatFilter)
	##cv.imshow('median blur', medianbl)
	cv.imshow('final', frame)
	cv.imshow('thresh', thresh)
	firstFrame = gray
	lastDelta = frameDelta


cap.release()
cv.destroyAllWindows()

