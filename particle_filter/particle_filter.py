import numpy as np
from scipy.spatial.transform import Rotation
import pdb
from scipy.stats import norm

from filterpy.monte_carlo import stratified_resample

class particleFilter:
	""" particle filter class to update and optimise camera poses """
	def __init__(self, n_particles=1000):
		# run particle initialisation
		self.n_particles = n_particles # number of particles

		# update noise from vo
		sigma_t = 3 # metres
		self.cov_t = [[sigma_t, 0, 0],[0, sigma_t, 0],[0, 0, sigma_t]]

		sigma_R = 5 # degrees
		self.cov_R = [[sigma_R, 0, 0],[0, sigma_R, 0],[0, 0, sigma_R]]

	def initialise_from_GPS(self, Easting, Northing, Heading):
		# distribute particle initially uniformly around GPS
		delta_GPS = 30
		particle_Easting = ((Easting + delta_GPS) - (Easting - delta_GPS)) * \
		  		np.random.random_sample((self.n_particles,1)) + (Easting - delta_GPS)

		particle_Northing = ((Northing + delta_GPS) - (Northing - delta_GPS)) * \
		  		np.random.random_sample((self.n_particles,1)) + (Northing - delta_GPS)

		# particle_Height
		max_h = 4
		min_h = 2

		particle_Height = (max_h - min_h) * \
		  		np.random.random_sample((self.n_particles,1)) + (min_h)

		# heading range
		delta_gamma = 20
		particle_gamma = ((Heading + delta_gamma) - (Heading - delta_gamma)) * \
		  		np.random.random_sample((self.n_particles,1)) + (Heading - delta_gamma)

		# beta range
		delta_beta = 5
		particle_beta = ((0 + delta_beta) - (0 - delta_beta)) * \
		  		np.random.random_sample((self.n_particles,1)) + (0 - delta_beta)
		
		# alpha range
		delta_alpha = 5
		particle_alpha = ((0 + delta_alpha) - (0 - delta_alpha)) * \
		  		np.random.random_sample((self.n_particles,1)) + (0 - delta_alpha)

		 # combine particles.
		self.particles_t = np.hstack((particle_Easting, particle_Height, particle_Northing))
		self.particles_R = Rotation.from_euler('xyz', 
					np.hstack((particle_alpha, particle_gamma, particle_beta)), degrees=True)

		# begin with uniform weights
		self.weights = (1/self.n_particles) * np.ones((self.n_particles))


	def initialise_along_the_highway(self, nav_file):
		""" particles will be distributed around the nav points """
		nav_coords = nav_file[['Easting','Northing','HEADING']]
		# draw from the nav_file
		nav_draws = nav_coords.sample(self.n_particles)
		
		delta_GPS = 5
		Easting_noise = (delta_GPS + delta_GPS) * np.random.random_sample((self.n_particles,1)) - delta_GPS
		particle_Easting = nav_draws['Easting'].to_numpy().reshape(self.n_particles,1) + Easting_noise

		Northing_noise = (delta_GPS + delta_GPS) * np.random.random_sample((self.n_particles,1)) - delta_GPS
		particle_Northing = nav_draws['Northing'].to_numpy().reshape(self.n_particles,1) + Northing_noise

		# particle_Height
		max_h = 4
		min_h = 2

		particle_Height = (max_h - min_h) * \
		  		np.random.random_sample((self.n_particles,1)) + (min_h)

		# losely point along the highway
		delta_gamma = 20
		Heading_noise = (delta_gamma + delta_gamma) * \
					np.random.random_sample((self.n_particles,1)) - delta_gamma
		particle_gamma = nav_draws['HEADING'].to_numpy().reshape(self.n_particles,1) + Heading_noise

		# beta range
		delta_beta = 5
		particle_beta = ((0 + delta_beta) - (0 - delta_beta)) * \
		  		np.random.random_sample((self.n_particles,1)) + (0 - delta_beta)
		
		# alpha range
		delta_alpha = 5
		particle_alpha = ((0 + delta_alpha) - (0 - delta_alpha)) * \
		  		np.random.random_sample((self.n_particles,1)) + (0 - delta_alpha)

		# combine particles.
		self.particles_t = np.hstack((particle_Easting, particle_Height, particle_Northing))
		self.particles_R = Rotation.from_euler('xyz', 
					np.hstack((particle_alpha, particle_gamma, particle_beta)), degrees=True)

		# begin with uniform weights
		self.weights = (1/self.n_particles) * np.ones((self.n_particles))	


	def update_particles_from_vo(self, R, t, scale=2.0):
		""" performs a rigid transformation t+1 = R_i t_i + t_i
		R_i+1 = R_i+1 * R_i """

		# particle update noise (white gaussian).
		# translation
		t_noise = np.random.multivariate_normal([0,0,0], self.cov_t, self.n_particles)
		self.particles_t = self.particles_t + scale * self.particles_R.apply(t) + t_noise

		# rotation
		R_noise = np.random.multivariate_normal([0,0,0], self.cov_R, self.n_particles)
		R_noise_RotObj = Rotation.from_euler('xyz', R_noise, degrees=True)
		self.particles_R = R_noise_RotObj * R * self.particles_R


	def update_weights_from_GPS(self, Easting, Northing):
		""" bit of a cheat but will update particles from a GPS reading """
		
		particle_x = self.particles_t[:,0].reshape(self.n_particles,1)
		particle_z = self.particles_t[:,2].reshape(self.n_particles,1)

		diff = np.hstack((particle_x - Easting, particle_z - Northing))
		dist = np.sum(np.abs(diff)**2,axis=-1)**(1./2)

		# use inverse quad
		weight_update = norm.pdf(dist, loc=0, scale=15)
		weights = np.multiply(self.weights, weight_update) # multiply by previous weight
		weights_norm = weights / np.sum(weights) # and normalise

		#pdb.set_trace()
		self.weights = weights_norm


	def resample(self):
		""" resampling """

		indices = stratified_resample(self.weights)		

		self.particles_t = self.particles_t[indices,:]
		self.particles_R = Rotation.from_euler('xyz',self.particles_R.as_euler('xyz')[indices,:])

		self.weights = (1/self.n_particles) * np.ones((self.n_particles))	



if __name__ == '__main__':
	
	PF = particleFilter(1)
	PF.initialise_from_GPS(1000,2000,1)

	print(PF.t)
	print(PF.R.as_euler('xyz'))