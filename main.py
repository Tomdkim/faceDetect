import json
from utils_image import (read_image, read_image_base64, save_image, draw_face, draw_box, draw_text)
import io
import os
from google.cloud import vision
import cv2
import numpy
from ast import literal_eval
import sys
import inspect

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="cred/service_account.json"

def main():

	inputfile = str(sys.argv[1])
	outputfile = "output of: " + inputfile

	myDict = {'chin_gnathion', 'forehead_glabella', 'left_ear_tragion', 'left_eye', 'left_eye_bottom_boundary', 'left_eye_left_corner', 'left_eye_pupil', 'left_eye_right_corner', 'left_eye_top_boundary', 'left_eyebrow_upper_midpoint', 'left_of_left_eyebrow', 'left_of_right_eyebrow', 'lower_lip', 'midpoint_between_eyes', 'mouth_center', 'mouth_left', 'mouth_right', 'nose_bottom_center', 'nose_bottom_left', 'nose_bottom_right', 'nose_tip', 'right_ear_tragion', 'right_eye', 'right_eye_bottom_boundary', 'right_eye_left_corner', 'right_eye_pupil', 'right_eye_right_corner', 'right_eye_top_boundary', 'right_eyebrow_upper_midpoint', 'right_of_left_eyebrow', 'right_of_right_eyebrow', 'upper_lip'}

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

		for attr in myDict:
			x = getattr(face._landmarks,attr)._position._x_coordinate
			y = getattr(face._landmarks,attr)._position._y_coordinate
			pt = (int(x),int(y))
			cv2.circle(a, pt, 3, (0,255,0))
		cv2.rectangle(a,literal_eval(vertices[0]),literal_eval(vertices[2]),(255,255,0),1)

		print('anger: {}'.format(face.emotions.anger))
		print('joy: {}'.format(face.emotions.joy))
		print('surprise: {}'.format(face.emotions.surprise))

		print('face bounds: {}'.format(','.join(vertices)))

	save_image(outputfile, a)
	
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