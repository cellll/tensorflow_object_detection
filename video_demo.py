import cv2
import tf_video
import sys

demo = tf_video.TF_DEMO()

videofile = sys.argv[1]
cap = cv2.VideoCapture(videofile)

while(True):
	ret, frame = cap.read()
	
	result = demo.detect(frame)

	cv2.imshow('', result)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

