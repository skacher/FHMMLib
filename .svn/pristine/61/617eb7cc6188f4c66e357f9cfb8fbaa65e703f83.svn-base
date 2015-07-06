/*
 * GMM.h
 *
 *  Created on: 11 мая 2015 г.
 *      Author: kacher
 */

#ifndef GMM_H_
#define GMM_H_

#include "HMMTypes.h"
#include "Config.h"

namespace HMM {

class GMM {
public:
	/*
	 * %GMM	Creates a Gaussian mixture model with specified architecture.
%
%	Description
%	 MIX = GMM(DIM, NCENTRES, COVARTYPE) takes the dimension of the space
%	DIM, the number of centres in the mixture model and the type of the
%	mixture model, and returns a data structure MIX. The mixture model
%	type defines the covariance structure of each component  Gaussian:
%	  'spherical' = single variance parameter for each component: stored as a vector
%	  'diag' = diagonal matrix for each component: stored as rows of a matrix
%	  'full' = full matrix for each component: stored as 3d array
%	  'ppca' = probabilistic PCA: stored as principal components (in a 3d array
%	    and associated variances and off-subspace noise
%	 MIX = GMM(DIM, NCENTRES, COVARTYPE, PPCA_DIM) also sets the
%	dimension of the PPCA sub-spaces: the default value is one.
%
%	The priors are initialised to equal values summing to one, and the
%	covariances are all the identity matrix (or equivalent).  The centres
%	are initialised randomly from a zero mean unit variance Gaussian.
%	This makes use of the MATLAB function RANDN and so the seed for the
%	random weight initialisation can be set using RANDN('STATE', S) where
%	S is the state value.
%
%	The fields in MIX are
%
%	  type = 'gmm'
%	  nin = the dimension of the space
%	  ncentres = number of mixture components
%	  covartype = string for type of variance model
%	  priors = mixing coefficients
%	  centres = means of Gaussians: stored as rows of a matrix
%	  covars = covariances of Gaussians
%	 The additional fields for mixtures of PPCA are
%	  U = principal component subspaces
%	  lambda = in-space covariances: stored as rows of a matrix
%	 The off-subspace noise is stored in COVARS.
	 *
	 */
#ifdef DEBUG_GMM
	GMM(size_t d, size_t M, const MatrixXf& data, const std::string& cov_type, const std::string& filename);
#else
	GMM(size_t d, size_t M, const MatrixXf& data, const std::string& cov_type);
#endif
	const MatrixXf& getCentres() const;
	const MultiD& getCovars() const;
private:
	MatrixXf Centres;
	MultiD Covars;
};

} /* namespace HMM */

#endif /* GMM_H_ */
