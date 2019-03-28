import cv2
import numpy as np


training_image = cv2.imread('test2.jpg')
training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(0)

while(True):

	ret, query_image = cap.read()
	query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

	path_front = "haarcascade_frontalface_alt2.xml"
	face_cascade = cv2.CascadeClassifier(path_front)

	faces = face_cascade.detectMultiScale(query_gray, scaleFactor=1.10,
										  minNeighbors=5, minSize=(40,40))

	for (x, y, w, h) in faces:
		# Cropped part of the video analyzed under the hood
		query_gray = query_gray[y-25:y+h+25, x-10:x+w+10]

	# Set the parameters of the ORB algorithm by specifying the maximum number of keypoints to locate and
	# the pyramid decimation ratio
	orb = cv2.ORB_create(1000, 2.0)

	# Find the keypoints in the gray scale training and query images and compute their ORB descriptor.
	# The None parameter is needed to indicate that we are not using a mask in either case.
	keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None)
	keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

	# Create a Brute Force Matcher object. Set crossCheck to True so that the BFMatcher will only return consistent
	# pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches.
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

	# Perform the matching between the ORB descriptors of the training image and the query image
	matches = bf.match(descriptors_train, descriptors_query)

	# The matches with shorter distance are the ones we want. So, we sort the matches according to distance
	matches = sorted(matches, key = lambda x : x.distance)

	# Connect the keypoints in the training image with their best matching keypoints in the query image.
	# The best matches correspond to the first elements in the sorted matches list, since they are the ones
	# with the shorter distance. We draw the first 300 mathces and use flags = 2 to plot the matching keypoints
	# without size or orientation.
	result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:300], query_gray, flags = 2)

	

	# Print the number of keypoints detected in the training image
	#print("Number of Keypoints Detected In The Training Image: ", len(keypoints_train))

	# Print the number of keypoints detected in the query image
	#print("Number of Keypoints Detected In The Query Image: ", len(keypoints_query))

	# Print total number of matching points between the training and query images
	#print("\nNumber of Matching Keypoints Between The Training and Query Images: ", len(matches))

	print(f"percent: {len(matches)/len(keypoints_query)} %", end="\r")


	cv2.imshow("Matching", result)
	cv2.imshow("Frame", query_image)

	ch = cv2.waitKey(1)
	# quite the loop if I tape on the letter q
	if ch & 0xFF == ord('q'):
		break


#cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()