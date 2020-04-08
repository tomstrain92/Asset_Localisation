import cv2
import matplotlib.pyplot as plt
import numpy as np

import pdb


def init_trajectory_and_gps_plot():

	fig, (VO_ax, gps_ax, error_ax) = plt.subplots(3,1)

	# setting titles and axis labels
	gps_ax.set_title('GPS')
	gps_ax.set_xlabel('Easting [m]')
	gps_ax.set_ylabel('Northing [m]')

	VO_ax.set_title('VO')
	VO_ax.set_xlabel('z [m]')
	VO_ax.set_ylabel('x [m]')

	error_ax.set_title("Odometry error")
	error_ax.set_xlabel("Image")
	error_ax.set_ylabel("Error [m]")
	errors = []


	# x goes to the right - is easier to visualise
	gps_ax.invert_yaxis()
	VO_ax.invert_yaxis()

	# axis equal
	gps_ax.set_aspect('equal')
	VO_ax.set_aspect('equal')

	return gps_ax, VO_ax, error_ax, errors


def draw_trajectory(traj_img, img_id, x, y, z, x_true, y_true, z_true, traj_img_size=800):

	half_traj_img_size = int(0.5*traj_img_size)

	draw_scale = 1

	draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
	true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
	cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
	cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
	# write text on traj_img
	cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
	cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
	# show 		
	cv2.imshow('Trajectory', traj_img)

	return traj_img


def draw_tracjectory_and_gps(VO_ax, gps_ax, error_ax, img_ind, plot_step, 
							 x, z, Easting, Northing, errors):

	# scatter new point
	VO_ax.scatter(z, x, marker='.', color="tab:red")

	# transform gps so it's relative to the first coord
	x_rel = Easting
	z_rel = Northing

	gps_ax.scatter(z_rel, x_rel, marker='.', color="tab:blue")

	# plot error
	error = np.linalg.norm([x_rel-x, z_rel-z])
	errors.append(error)
	error_diff = np.diff(errors)
	error_ax.plot(range(0, plot_step * len(error_diff), plot_step), error_diff, color='tab:green')
	error_ax.plot(range(0, plot_step * len(errors[1:]), plot_step), errors[1:], color='tab:orange')

	error_ax.legend(['Local','Global'])

	plt.pause(0.00001)

	return VO_ax, gps_ax, error_ax, errors


def relative_gps_coord(Easting, Northing, Easting_begin, Northing_begin, Heading_begin):


	theta = np.deg2rad(Heading_begin) #defined from y axis so change to x.
	Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
				   [np.sin(theta),  np.cos(theta), 0],
				   [0, 0, 1]])
	# translation vector is just back to the first gps 
	t = np.array([Easting_begin, Northing_begin, 0])
	# coordinate of current gps
	Xw = np.array([Easting, Northing, 0])
	#print(Xw-t)
	print(Xw-t)

	# transform to vehicle
	Xv = np.matmul(Rz, (Xw - t))

	x_rel = Xv[0]
	z_rel = Xv[1]

	return x_rel, z_rel


