import json
import io
import os
from google.cloud import vision
import cv2
import numpy
from ast import literal_eval
import sys
import inspect
from base64 import b64encode

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="cred/service_account.json"

def main():

	inputfile = str(sys.argv[1])
	outputfile = "output of: " + inputfile

	myDict = {'chin_gnathion', 'left_ear_tragion', 'left_eye_pupil', 'mouth_center', 'right_ear_tragion', 'right_eye_pupil'}

	midpointeyesDict = {'forehead_glabella', 'midpoint_between_eyes','nose_tip'}
	lefteyeDict = {'left_eye', 'left_eye_bottom_boundary', 'left_eye_left_corner', 'left_eye_right_corner', 'left_eye_top_boundary'}
	righteyeDict = {'right_eye', 'right_eye_bottom_boundary', 'right_eye_left_corner', 'right_eye_right_corner', 'right_eye_top_boundary'}
	lefteyebrowDict = {'left_eyebrow_upper_midpoint', 'left_of_left_eyebrow', 'right_of_left_eyebrow'}
	righteyebrowDict = {'left_of_right_eyebrow', 'right_eyebrow_upper_midpoint', 'right_of_right_eyebrow'}
	noseDict = {'nose_tip','nose_bottom_right','nose_bottom_center','nose_bottom_left'}
	mouseDict = {'lower_lip', 'mouth_left', 'upper_lip', 'mouth_right'}
	list_of_dict = [midpointeyesDict,lefteyeDict,righteyeDict,lefteyebrowDict,righteyebrowDict,noseDict,mouseDict]

	vision_client = vision.Client()
	body = make_request(inputfile)

	with io.open(inputfile, 'rb') as image_file:
		content = image_file.read()

	response = vision_client.image(content=content)
	faces = response.detect_faces()

	im = cv2.imread(inputfile)
	a = numpy.asarray(im)

	print('Faces:')

	for face in faces:

		vertices = (['({},{})'.format(bound.x_coordinate, bound.y_coordinate)
					for bound in face.bounds.vertices])

		for dictionary in list_of_dict:
			points = []
			target = 0
			for attr in dictionary:
				x = getattr(face._landmarks,attr)._position._x_coordinate
				y = getattr(face._landmarks,attr)._position._y_coordinate
				pt = (int(x),int(y))
				points.append(pt)
				cv2.circle(a, pt, 1, (0,255,0))
			length = len(points)
			while (target < length):
				if (target == length - 1):
					cv2.line(a, points[target], points[0], (0,255,0), thickness=1)
				else:
					cv2.line(a, points[target], points[target+1], (0,255,0), thickness=1)
				target += 1


		# for attr in myDict:
		# 	x = getattr(face._landmarks,attr)._position._x_coordinate
		# 	y = getattr(face._landmarks,attr)._position._y_coordinate
		# 	pt = (int(x),int(y))
		# 	points.append(pt)
		# 	cv2.circle(a, pt, 3, (0,255,0))

		# length = len(points)
		# count = 0
		# target = 0
		# while (target < length):
		# 	count = target + 1
		# 	while (count < length):
		# 		cv2.line(a, points[target], points[count], (0,255,0), thickness=1)
		# 		count += 1
		# 	target += 1


		cv2.rectangle(a,literal_eval(vertices[0]),literal_eval(vertices[2]),(255,255,0),1)

		print('anger: {}'.format(face.emotions.anger))
		print('joy: {}'.format(face.emotions.joy))
		print('surprise: {}'.format(face.emotions.surprise))

		print('face bounds: {}'.format(','.join(vertices)))

	cv2.imwrite(outputfile, a)

def read_image_base64(filename):
	with open(filename, 'rb') as f:
		return b64encode(f.read())
	
def make_request(inputfile):
	""" Create a request batch (one file at a time) """
	return {
		"requests":[
			{
				"image":{
	    				"content": read_image_base64(inputfile)
	    			},
				"features": [
					{
						"type":"LABEL_DETECTION",
      						"maxResults": 10
      					},
      					{
      						"type":"TEXT_DETECTION",
      						"maxResults": 10
      					},
      					{
      						"type":"FACE_DETECTION",
      						"maxResults": 20
      					}
      				]
			}
		]
	}

if __name__ == '__main__':
	main()