/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    num_particles = 100;

    particles.resize(num_particles);

    // randome generator
    std::mt19937 gen;

    // create normal distributions using mean (GPS data) and standard deviations
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    // sample from distribution and add to the collection of particles
    for (int i=0; i<num_particles; ++i) {
        Particle sample;
        sample.x = dist_x(gen);
        sample.y = dist_y(gen);
        sample.theta = dist_theta(gen);
        sample.weight = 1;
        particles.push_back(sample);

    }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    std::mt19937 gen;

	for (int i=0; i<num_particles; ++i) {
		// calculate mean of prediction
		double xt = particles[i].x;
		double yt = particles[i].y;
		double thetat = particles[i].theta;

		double x_mean = xt + velocity / yaw_rate * (sin(thetat + yaw_rate * delta_t) - sin(thetat));
		double y_mean = yt + velocity / yaw_rate * (cos(thetat) - cos(thetat + yaw_rate * delta_t));
		double theta_mean = thetat + yaw_rate * delta_t;

		// create normal distribution 
		normal_distribution<double> dist_x(x_mean, std_pos[0]);
		normal_distribution<double> dist_y(y_mean, std_pos[1]);
		normal_distribution<double> dist_theta(theta_mean, std_pos[2]);

		// predict with uncertainty
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, const std::vector<Map::single_landmark_s>& landmark_list, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i=0; i<predicted.size(); ++i) {

		// calculate initial distance and assign id to first landmark
		double min_distance = sqrt(pow((predicted[i].x-landmark_list[0].x_f),2)+pow((predicted[i].y-landmark_list[0].y_f),2));
		predicted[i].id = landmark_list[0].id_i;

		// loop through all the rest of landmarks and calculate distance
		for (int j=1; j<landmark_list.size(); ++j) {
			double distance_to_landmark = sqrt(pow((predicted[i].x-landmark_list[j].x_f),2)+pow((predicted[i].y-landmark_list[j].y_f),2));
			if (distance_to_landmark < min_distance && distance_to_landmark < sensor_range) {
				min_distance = distance_to_landmark;
				predicted[i].id = landmark_list[j].id_i;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i=0; i<num_particles; ++i) {
	    // transform observations in vehicle coordinate system to map coordinate system
	    std::vector<LandmarkObs> predicted_obs;
	    predicted_obs.resize(observations.size());
	    for (int j=0; j<observations.size(); ++j) {
		    double x_map = particles[i].x + cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y;
		    double y_map = particles[i].y + sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y;
		    predicted_obs[j].x = x_map;
		    predicted_obs[j].y = y_map;
	    }

	    // association
	    dataAssociation(predicted_obs, map_landmarks.landmark_list, sensor_range);

	    // update weights
	    for (int j=0; j<observations.size(); ++j) {
	    	double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	    	double x = predicted_obs[j].x;
	    	double mu_x = map_landmarks.landmark_list[predicted_obs[j].id].x_f;
	    	double y = predicted_obs[j].y;
	    	double mu_y = map_landmarks.landmark_list[predicted_obs[j].id].y_f;

	    	double exponent = pow(x-mu_x,2) / (2 * pow(std_landmark[0],2)) + pow(y-mu_y,2) / (2 * pow(std_landmark[1],2));

	    	particles[i].weight *= gauss_norm * exp(-exponent);
	    }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	//  std::default_random_engine generator;
    //std::discrete_distribution<int> distribution {2,2,1,1,2,2,1,1,2,2};
    std::mt19937 gen;

    // create a vector of weights
    std::vector<double> weights;
    for (int i=0;i<particles.size(); ++i) {
    	weights.push_back(particles[i].weight);
    }

	// create distribution
    std::discrete_distribution<> dist_w(weights.begin(), weights.end());

    //resample
    std::vector<Particle> new_particles;
    new_particles.resize(particles.size());
    for (int i=0;i<particles.size(); ++i) {
    	new_particles.push_back(particles[dist_w(gen)]);
    }

    // replace old particles
	particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
