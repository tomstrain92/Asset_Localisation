
import matplotlib.pyplot as plt
from PIL import Image
import math


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