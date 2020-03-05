import os, sys
import pandas as pd
pd.options.mode.chained_assignment = None
from simpledbf import Dbf5
import numpy as np
from plotting import *
from shutil import copy2
import glob


class Survey:
	""" Survey Class. Members:
	nav_file - pandas dataframe of the nav_file.dbf file.
	data_dir - the path the asset inventory and nav_file folder. """

	def __init__(self, data_dir):

		self.data_dir = data_dir
		self.nav_file = self.load_nav_file()
	

	def load_assets(self, asset_type):
		"""loads the asset file of asset type"""
		asset_file = glob.glob(os.path.join(self.data_dir,"Assets","*{}*.dbf".format(asset_type)))
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
		""" loads nav file into a pandas DataFrame"""
		dbf = Dbf5(os.path.join(self.data_dir,"Nav","Points.dbf"))
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
		np_coordinate = coordinate.to_numpy().reshape(-1)

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


	def find_close_assets(self, assets, image=None, PCDATE=None, PCTIME=None, max_dist=40):
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
		for (ind, asset) in close_assets.iterrows():
			# compute coordinate relative to the vehicle.
			vehicle_coordinates = self.coordinate_relative_to_vehicle(asset, image)
			close_assets.loc[ind, "vehicle_x"] = vehicle_coordinates[0]
			close_assets.loc[ind, "vehicle_y"] = vehicle_coordinates[1]
			close_assets.loc[ind, "vehicle_z"] = vehicle_coordinates[2]

		return close_assets


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




