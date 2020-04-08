import numpy as np
import cv2
import argparse
import os, glob
import sys
sys.path.insert(0,'..')
from survey import Survey 
from imantics import Polygons, Mask, BBox
import pdb

from image_processing import *


def blue_mask(hsv):

	lower_blue = np.array([70, 100, 2])
	upper_blue = np.array([160,255,255])
	mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

	return mask_blue


def segment(img):
	
	# make color HSV image - second image for color mask recognition
    # Convert BGR to HSV
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define the list of boundaries (lower and upper color space for
    # HSV
    # mask for red color consist from 2 parts (lower mask and upper mask)
    # blue color mask

	mask_blue = blue_mask(hsv)

	# Range for lower red
	lower_red = np.array([0,50,0])
	upper_red = np.array([10,255,100])
	mask_red_1 = cv2.inRange(hsv, lower_red, upper_red)
	# Range for upper range
	lower_red = np.array([160,50,0])
	upper_red = np.array([180,255,100])
	mask_red_2 = cv2.inRange(hsv,lower_red,upper_red)#

	# Generating the final mask to detect red color
	mask_red = mask_red_1+mask_red_2

    # join all masks
    # could be better to join yellow and red mask first  - it can helps to detect
    # autumn trees and delete some amount of garbage, but this is TODO next  
	mask = mask_blue

	# blue and morph mask
	mask = cv2.blur(mask,(2,2))
	kernel = np.ones((5,5),np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	# finding boxes
	_, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	bounding_boxes = np.empty((0,4), dtype=np.uint8)
	for con in contours:
		rect = np.array(cv2.boundingRect(con)).reshape(1,4)
		bounding_boxes = np.append(bounding_boxes,rect,0)
	
	# turn to a BGR image for plotting
	mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #hsv_out = cv2.bitwise_and(hsv, hsv, mask = mask)

	return mask, mask_img, bounding_boxes


def clean_boxes(boxes, mask):

	clean_boxes = np.empty((0,4), dtype=np.uint8)
	# first perform NMS
	bounding_boxes_NMS = non_max_suppression_fast(boxes, 0.1)
	for box in bounding_boxes_NMS:
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		if not y < 10 and not y + h > 610:
			if w*h > 50:
				print(x,y)
				box_pixels = mask[y:y+h, x:x+w] # X AND Y ARE FLIPPED HERE!
				extent = np.sum(box_pixels>200) / (box_pixels.shape[0] * box_pixels.shape[1])
				if extent > 0.1:
					clean_boxes = np.append(clean_boxes, [box], axis=0)

	return clean_boxes


def main(args):


	survey = Survey(args.data_dir, args.road)

	Easting_begin, Northing_begin, Heading_begin = survey.begin_image_sequence(index=11000, resize=0.25)


	resize = 0.25
	cv2.namedWindow("RGB")

	for i in range(10000):
		img, Easting, Northing = survey.next()
		# resize
		mask, mask_img, bounding_boxes = segment(img)

		# showing before processing
		for box in bounding_boxes:
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]
			if w*h > 100:
				img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		# now clean
		bounding_boxes = clean_boxes(bounding_boxes, mask)
		for box in bounding_boxes:
			x = box[0]
			y = box[1]
			w = box[2]
			h = box[3]
			img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			#text = "{:.2f}".format(extent)
			#cv2.putText(img, text, (int(x+w/2), int(y+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

		numpy_horizontal = np.hstack((img, mask_img))

		#cv2.imshow("RGB", img)
		cv2.imshow("RGB",numpy_horizontal)
		cv2.waitKey(1)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Asset location tool')
	parser.add_argument("road", type=str,
						help="road/highway to split")
	parser.add_argument("--data_dir", "-d", type=str, default="/media/tom/Elements", 
						help="top level data directory")
	parser.add_argument("--set_num", type=int, default=0, help="folder of images")

	args = parser.parse_args()
	
	main(args)