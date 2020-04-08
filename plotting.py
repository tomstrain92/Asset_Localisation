import matplotlib.pyplot as plt
from PIL import Image
import math
from survey import * 

from convertbng.util import convert_bng, convert_lonlat

import shapefile as shp  # Requires the pyshp package
import geopandas
import os

import pdb


class surveyMap:
	""" using the shape files to quickly update the plots 
	this is built upon the survey class"""
	def __init__(self, survey):
		print("loading map...")
		self.create_map_from_shape_file(survey)
		self.data_dir = survey.data_dir
		self.road = survey.road


	def create_map_from_shape_file(self, survey):
		""" create map of road network """
		df = geopandas.read_file(survey.shape_file)
		self.df_road = df[(df.descriptiv == "Road Or Track") |
	    			 (df.descriptiv == "(1:Road Or Track)")]
		#self.df_road = df
		#fig, (self.map_ax, self.map_zoomed_ax)= plt.subplots(1,2)
		fig = plt.figure()
		self.map_ax = fig.gca()

		fig = plt.figure()
		self.map_zoomed_ax = fig.gca()

		self.df_road.plot(ax=self.map_ax)
		self.df_road.plot(ax=self.map_zoomed_ax)

		# create centroids - this will be used for killing particles that are far away from the road.
		centroids = df.geometry.centroid
		self.road_geometry = pd.concat([centroids.x.reset_index(drop=True), centroids.y.reset_index(drop=True)], axis=1)
		self.road_geometry.columns = ["Easting","Northing"]

		# so there's something to remove
		self.gps_scatter = self.map_ax.scatter(1,1)
		self.particle_scatter = self.map_ax.scatter(1,1)

		self.gps_scatter_zoomed = self.map_zoomed_ax.scatter(1,1)
		self.particle_scatter_zoomed = self.map_zoomed_ax.scatter(1,1)

		self.exp_scatter = self.map_ax.scatter(1,1)
		self.exp_scatter_zoomed = self.map_zoomed_ax.scatter(1,1)


	def plot_assets_from_shape_files(self, asset_types):
		""" plots assets from their shape file"""
		for asset_type in asset_types:
			asset_shape_file = os.path.join(self.data_dir, self.road, "Inventory", "{}.shp".format(asset_type))
			df_asset = geopandas.read_file(asset_shape_file)
			df_asset.plot(ax=self.map_zoomed_ax, color='white', edgecolor='black')


	def update_GPS_trajectory(self, Easting, Northing, map_buffer=200):
		""" updates plot with gps position """
		self.gps_scatter.remove()
		self.gps_scatter_zoomed.remove()
		self.gps_scatter = self.map_ax.scatter(Easting, Northing, color='r')
		self.gps_scatter_zoomed = self.map_zoomed_ax.scatter(Easting, Northing, color='r')

		self.map_zoomed_ax.set(xlim=(Easting-map_buffer, Easting+map_buffer), ylim=(Northing-map_buffer, Northing+map_buffer))
		

	def update_particles(self, pf, map_buffer = 50):
		""" plots each of the particles and rescales map"""
		
		self.particle_scatter.remove()
		self.particle_scatter_zoomed.remove()

		self.particle_scatter = self.map_ax.scatter(pf.particles_t[:,0], pf.particles_t[:,2], color='g', alpha=0.1)
		self.particle_scatter_zoomed = self.map_zoomed_ax.scatter(pf.particles_t[:,0], pf.particles_t[:,2], color='g', alpha=0.1)

		Easting_min = pf.particles_t[:,0].min() - map_buffer
		Easting_max = pf.particles_t[:,0].max() + map_buffer

		Northing_min = pf.particles_t[:,2].min() - map_buffer
		Northing_max = pf.particles_t[:,2].max() + map_buffer

		self.map_ax.set(xlim=(Easting_min, Easting_max), ylim=(Northing_min, Northing_max))

		# Expectation
		Easting_exp = pf.particles_t[:,0].dot(pf.weights)
		Northing_exp = pf.particles_t[:,2].dot(pf.weights)
 		
		self.exp_scatter.remove()
		self.exp_scatter_zoomed.remove()

		self.exp_scatter = self.map_ax.scatter(Easting_exp, Northing_exp, color='b')
		self.exp_scatter_zoomed = self.map_zoomed_ax.scatter(Easting_exp, Northing_exp, color='b')


	def distance_to_road_edge(self, point):
		""" returns the smallest distance to the geometry points - if this is small then likely driving off the road """
		diffs = self.road_geometry.to_numpy() - point
		return np.linalg.norm(diffs, axis=1).min()
		



def show_asset_images(asset_image_list, asset_type=None, asset_id=None):
	n_images = len(asset_image_list)
	cols = math.ceil(n_images/2)
	rows = 2
	print(n_images)
	fig = plt.figure()
	if asset_type is not None:
		fig.suptitle("{:s}_{:d}".format(asset_type,asset_id))
		file_name = ("{:s}_{:d}".format(asset_type,asset_id))

	for ind, image in enumerate(asset_image_list):
		img = Image.open(image)
		print(rows * cols, ind+1)
		fig.add_subplot(rows, cols, ind+1)
		plt.imshow(img)
		plt.axis('off')

	plt.savefig("asset_images.png")


def plot_vehicle_headings(nav_file, assets=None, asset_coord=None):

	for (ind, image) in nav_file.iterrows():

		plt.plot(image['Easting'],image['Northing'],'ro')
		# plotting heading
		theta = np.deg2rad(image['HEADING'])
		arm = 1
		x_arm = float(image['Easting']) + arm * np.sin(theta)
		y_arm = float(image['Northing']) + arm * np.cos(theta)
		plt.plot([float(image['Easting']), x_arm], [float(image['Northing']), y_arm],'k')
		

	asset_coords = assets[['Easting','Northing']].to_numpy()
	plt.plot(asset_coords[:,0], asset_coords[:,1],'o')

	plt.grid()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.savefig('headings')



def plot_vehicle_and_assets(image, assets):
	
	print("plotting")
	plt.plot(image['Easting'],image['Northing'],'o')
	asset_coords = assets[['Easting','Northing']].to_numpy()
	plt.plot(asset_coords[:,0], asset_coords[:,1],'o')
	# plotting heading
	theta = np.deg2rad(image['HEADING'])
	arm = 1
	x_arm = float(image['Easting']) + arm * np.cos(theta)
	y_arm = float(image['Northing']) + arm * np.sin(theta)
	plt.plot([float(image['Easting']), x_arm], [float(image['Northing']), y_arm])

	plt.savefig('heading.png')


if __name__ == "__main__":

	survey = Survey("/media/tom/Elements", "M69")
	survey_map = Map(survey)

	#survey_map.ax.set_aspect('auto')

	for ind,nav in survey.nav_file.iterrows():
		x = nav.Easting
		y = nav.Northing
		survey_map.update_plot_GPS_trajectory([x, y])
		survey_map.ax.set(xlim=(x-50, x+50), ylim=(y-50, y+50))
		plt.pause(0.001)
