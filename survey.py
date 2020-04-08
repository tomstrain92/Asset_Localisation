import os, sys
import pandas as pd
pd.options.mode.chained_assignment = None
from simpledbf import Dbf5
import numpy as np
from plotting import *
from shutil import copy2
import glob
import time
import sqlite3
from PIL import Image
import matplotlib.image as mpimg
import cv2

import pdb

class Survey:
	""" Survey Class. """

	def __init__(self, data_dir, road):

		self.data_dir = data_dir
		self.road = road
		self.nav_file = self.load_nav_file()
		self.shape_file = os.path.join(data_dir, road, "Nav", "{}_Map.shp".format(road))


	def begin_image_sequence(self, index=None, PCDATE=None, PCTIME=None, resize=1):
		""" starts an image nav sequence and returns the gps-IMU info.. largely for plotting """
		if index is not None:
			self.image_ind = index
		else:
			self.image_ind = self.nav_file[(self.nav_file['PCDATE']==PCDATE) & 
								      (self.nav_file['PCTIME']==PCTIME)]['index']

		nav = self.nav_file[(self.nav_file['index']==self.image_ind)]
		self.resize = resize # make images smaller.

		return float(nav['Easting']), float(nav['Northing']), float(nav['HEADING'])


	def next(self):
		""" returns the img and gps tag of the next nav point in sequence """
		nav = self.nav_file[self.nav_file['index']==int(self.image_ind)]
		img = self.load_image(nav['PCDATE'], nav['PCTIME'], format='opencv', channels='RGB', resize=self.resize)
		self.image_ind = self.image_ind + 1

		return img, float(nav['Easting']), float(nav['Northing'])


	def load_assets(self, asset_type):
		"""loads the asset file of asset type"""
		asset_file = glob.glob(os.path.join(self.data_dir,self.road,"Inventory","*{}*.dbf".format(asset_type)))
		dbf = Dbf5(asset_file[0])
		df = dbf.to_dataframe()
		# Define Easting and Northing
		df["Easting"] = df["XCOORD"]
		df["Northing"] = df["YCOORD"]
		df["SOURCE_ID"] = df["SOURCE_ID"].astype(str).astype(int)
		# if no height info - add it
		if "MOUNTING_H" not in df.columns:
			df["MOUNTING_H"] = 0
		# add asset type to class
		self.asset_type = asset_type
		return df


	def load_nav_file(self):
		""" loads nav file from database"""
		print('loading nav file database')
		start = time.time()
		conn = sqlite3.connect(os.path.join(self.data_dir, 'nav_files.sqlite'), sqlite3.PARSE_COLNAMES) 
		c = conn.cursor()
		c.execute("SELECT name FROM sqlite_master WHERE type='table';")
		print(c.fetchall())
		query = 'SELECT * FROM {}'.format(self.road)
		# execture sql query
		data = c.execute(query) 
		col_names = [desc[0] for desc in c.description]	
		# convert to dataframe
		nav_file = pd.DataFrame(data, columns=col_names)
		finish = time.time()
		print("nav file loaded in {:2f} seconds".format(finish - start))
		
		return nav_file


	def load_image(self, PCDATE, PCTIME, camera=2, format='PIL', channels='RGB', resize=None):
		''' returns forward facing image defined by PCDATE and PCTIME as a "format" image '''
		image_file = os.path.join(self.data_dir, self.road, 'Images',
							'{:d}_{:d}_{:d}.jpg'.format(camera, int(PCDATE), int(PCTIME)))
		# chose correct function. 
		if format == 'PIL':
			if channels == 'G':
				return Image.open(image_file,0)
			else:
				return Image.open(image_file)
		elif format == 'matplotlib':
			return mpimg.imread(image_file)
		elif format == 'opencv':
			if channels == 'G':
				img = cv2.imread(image_file,0)
			else:
				img = cv2.imread(image_file)
				h,w,c = img.shape
			if resize is not None:
				img = cv2.resize(img, (int(w * resize), int(h * resize)))
			return img
		else:
			print('Error, incorrect image format given')



	def load_nav_file_from_disk(self):
		""" loads nav file into a pandas DataFrame"""
		dbf = Dbf5(os.path.join(self.data_dir,"Nav","{}.dbf".format(self.road)))
		df = dbf.to_dataframe()
		# PCDATE and PCTIME read in as objects
		df["PCDATE"] = df["PCDATE"].astype(str).astype(int)
		df["PCTIME"] = df["PCTIME"].astype(str).astype(int)
		# Define Easting and Northing
		df["Easting"] = df["XCOORD"]
		df["Northing"] = df["YCOORD"]

		file_names = []
		for (ind, image) in df.iterrows():
			file_names.append(os.path.join(self.data_dir, "Images", "1_{}_{}.jpg".format(image["PCDATE"], image["PCTIME"])))

		df['file_name'] = file_names
		return(df)


	def coordinate_relative_to_vehicle(self, coordinate, image=None, PCDATE=None, PCTIME=None):
		"""computes vector relative the vehicle, from the forward facing image
		coordinate is [Easting, Northing, Height]"""

		if image is None:
			image = self.nav_file[(self.nav_file["PCDATE"]==PCDATE) & 
							  (self.nav_file["PCTIME"]==PCTIME)]

		# turn coordinate into numpy arrray
		coordinate = coordinate[["Easting","Northing","MOUNTING_H"]]
		np_coordinate = coordinate.to_numpy().reshape(-1).astype(float)

		# camera coordinate as numpy array
		np_camera_position = np.array([float(image["Easting"]),
									   float(image["Northing"]),
									   2.5])

		# vector from coordinate to the camera.
		np_coordinate_vehicle = np_coordinate - np_camera_position
		# now rotate this vector by the heading angle (in radians)
		theta = np.deg2rad(float(image["HEADING"]))
		# create 3D rotation matrix as https://en.wikipedia.org/wiki/Rotation_matrix
		Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
					   [np.sin(theta), np.cos(theta), 0],
					   [0, 0, 1]])
		# matrix muliply the rotation matrix by the vector
		np_rotated_coordinate_vehicle = Rz @ np_coordinate_vehicle
		return [np_rotated_coordinate_vehicle[1], np_rotated_coordinate_vehicle[0], np_rotated_coordinate_vehicle[2]]


	def find_close_assets(self, assets, image=None, PCDATE=None, PCTIME=None, max_dist=70):
		"""finds assets within a radius of an image"""
		# if PCDATE given instead..
		if image is None:
			image = self.nav_file[(self.nav_file["PCDATE"]==PCDATE) & 
							  (self.nav_file["PCTIME"]==PCTIME)]
		# distance from the image
		delta_E = assets["Easting"] - float(image["Easting"])
		delta_N = assets["Northing"] - float(image["Northing"])
		distance = np.sqrt((delta_E * delta_E) + (delta_N * delta_N))

		# find any assets within max_dist
		close_assets = assets[distance < max_dist]
		# create empty columns
		close_assets["vehicle_x"] = np.nan
		close_assets["vehicle_y"] = np.nan
		close_assets["vehicle_z"] = np.nan

		# for each of these close_assets now convert to the frame of the vehicle
		infront = False
		for (ind, asset) in close_assets.iterrows():
			# compute coordinate relative to the vehicle.
			vehicle_coordinates = self.coordinate_relative_to_vehicle(asset, image)
			close_assets.loc[ind, "vehicle_x"] = vehicle_coordinates[0]
			close_assets.loc[ind, "vehicle_y"] = vehicle_coordinates[1]
			close_assets.loc[ind, "vehicle_z"] = vehicle_coordinates[2]

			if vehicle_coordinates[0] > 20: 
				infront = True

		return close_assets, infront 


	def find_close_images(self, asset, max_dist=30):
		"""finds images within a radius of max dist of asset"""
		delta_E = self.nav_file["Easting"] - float(asset["Easting"])
		delta_N = self.nav_file["Northing"] - float(asset["Northing"])
		distance = np.sqrt((delta_E * delta_E) + (delta_N * delta_N))

		close_images = self.nav_file[distance < max_dist]
		return close_images


	def create_asset_images(self, assets, asset_id, save_images=False):
		"""load a series of images from the left facing camera. The images are
		taken 12-28m infront of the asset """
		asset = assets[assets["SOURCE_ID"] == asset_id]
		asset_coordinate = asset[["Easting","Northing","MOUNTING_H"]]
		# first find images that are close to the asset
		close_images = self.find_close_images(asset)
		# then loop through and check if in range, and to the right of the asset
		asset_image_list = []

		# the images will be saved create the folder
		if save_images:
			# create new folder
			out_path = os.path.join("Embankment_Asset_Images", self.asset_type, str(asset_id))
			if not os.path.exists(out_path): os.makedirs(out_path)

		for (ind, image) in close_images.iterrows():
			vehicle_coordinates = self.coordinate_relative_to_vehicle(float(image["PCDATE"]),
																	  float(image["PCTIME"]),
																	  asset)
			# if behind and to the right of the asset add image file
			if (vehicle_coordinates[0] < 18 and vehicle_coordinates[0] > 10 and vehicle_coordinates[1] < 0):
				# create image file and check it exists
				image_file = os.path.join(self.data_dir, "Images",
								"1_{}_{}.jpg".format(image["PCDATE"], image["PCTIME"]))
				# if it exists, add to list and save if required
				if os.path.exists(image_file):
					asset_image_list.append(image_file)
					# save image
					if save_images: copy2(image_file, out_path)

		return asset_image_list


def resize_images(data_dir, road, PCDATE, PCTIME, scale=0.25):
	"""function to resize the rgb and depth images if they exist"""

	# ensure that the folders exists
	resized_image_folder = os.path.join(data_dir, road, "Images_resized")
	resized_depth_folder = os.path.join(data_dir, road, "Depth_resized")
	if not os.path.exists(resized_image_folder): os.makedirs(resized_image_folder)
	if not os.path.exists(resized_depth_folder): os.makedirs(resized_depth_folder)


	# these are the resized files. 
	resized_image_path = os.path.join(resized_image_folder, 
		"{:d}_{:d}_{:d}.png".format(2, PCDATE, PCTIME))
	resized_depth_path = os.path.join(resized_depth_folder,
		"{:d}_{:d}_{:d}.png".format(2, PCDATE, PCTIME))

	# if the file doesn't exist then resize
	if not os.path.exists(resized_image_path):
		print("resizing")
		# original image and depth
		image_path = os.path.join(data_dir, road, "Images",
				"{:d}_{:d}_{:d}.jpg".format(2, PCDATE, PCTIME))
		image = Image.open(image_path)
			
		depth_path = os.path.join(data_dir, road, "Depth",
				"{:d}_{:d}_{:d}.png".format(2, PCDATE, PCTIME)) # png for matplotlib
		depth = Image.open(depth_path)

		# get width and height and resize
		width, height = image.size
		width_new = int(width * scale)
		height_new = int(height * scale)

		# resize
		image_resized = image.resize((width_new, height_new))
		depth_resized = depth.resize((width_new, height_new))

		# save
		image_resized.save(resized_image_path)
		depth_resized.save(resized_depth_path)


if __name__ == "__main__":
	survey = Survey('/media/tom/Elements','M45')
	survey.begin_image_sequence(index=200)

	print(survey.nav_file[['PCDATE','PCTIME']].head())

	plt.figure()
	for i in range(100):
		img = survey.next(resize=0.5)
		plt.imshow(img)
		plt.pause(0.1)
