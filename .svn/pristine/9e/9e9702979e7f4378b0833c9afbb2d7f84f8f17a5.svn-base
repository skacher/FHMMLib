/*
 * GMM.cpp
 *
 *  Created on: 11 мая 2015 г.
 *      Author: kacher
 */

//#include <random>

#include <limits>

#include "../../base/GMM.h"
#include "../../base/MatlabUtils.h"
#include "../../base/KMeansInit.h"

#include "../../io/Util.h"

#ifdef DEBUG_GMM
#include <iostream>
#endif

namespace HMM {


#ifdef DEBUG_GMM
GMM::GMM(size_t dim, size_t ncentres, const MatrixXf& data, const std::string& covar_type, const std::string& filename) {
#else
GMM::GMM(size_t dim, size_t ncentres, const MatrixXf& data, const std::string& covar_type) {
#endif

	//% Initialise centres
#ifdef DEBUG_GMM
	MatrixXf centres = Read2DMatrix(filename); //mix.centres = randn(mix.ncentres, mix.nin);
#else
	MatrixXf centres = randn(ncentres, dim); //mix.centres = randn(mix.ncentres, mix.nin);
#endif

	MultiD covars = MultiD(1,1);

	if (covar_type == "spherical") {
		covars(0,0) = MatrixXf::Ones(1, ncentres);
	}
	else if(covar_type == "diag") {
		covars(0,0) = MatrixXf::Ones(ncentres, dim);
	}
	else { // full
		covars = repmat<MatrixXf, MultiD>(MatrixXf::Identity(dim, dim), 1, 1, ncentres, 1);
	}
	////////////////////////////////////////////////////////////////////////////
	//HERE FINISH GMM.M
	////////////////////////////////////////////////////////////////////////////

	MatrixXf x = data;


	double gmm_width = 1.0;

#ifdef DEBUG_GMM
	KMeansInit kmi(centres, x, filename + "_perm"); //[mix.centres, options, post] = kmeansInit(mix.centres, x, options);
#else
	KMeansInit kmi(centres, x); //[mix.centres, options, post] = kmeansInit(mix.centres, x, options);
#endif
	Centres = kmi.getCentres();
	MatrixXf post = kmi.getPost();

	MatrixXf cluster_sizes;
	MatrixXf sum_cluster_sizes;
	MatrixXf tmp_index;
	max_with_index( post.colwise().sum(), 1, cluster_sizes, tmp_index );
	max_with_index(cluster_sizes, 1, sum_cluster_sizes, tmp_index );

	if (covar_type == "spherical") {
		if (ncentres > 1) {
			/*
			 *       % Determine widths as distance to nearest centre
  	   	   	   	   % (or a constant if this is zero)
			 */
			MatrixXf cdist = dist2(Centres, Centres);
			cdist += (MatrixXf::Ones(ncentres, 1) * std::numeric_limits<double>::max()).asDiagonal();
			MatrixXf tmp_index;
			MatrixXf tmp_covars;
			min_with_index(cdist, 1, tmp_covars, tmp_index);

			MatrixXf tmp = (tmp_covars.array() < std::numeric_limits<double>::epsilon( )).cast<float>();

		    tmp_covars +=  gmm_width * ( tmp );
		    covars(0,0) = tmp_covars;
		}
		else {

			/*
			 * % Just use variance of all data points averaged over all
      	  	  % dimensions
   	   	   */

			covars(0,0) = (varaince<MatrixXf>(x)).rowwise().mean();
			//print4DMatrix(covars);
		}
	}
	else if(covar_type == "diag") {
		for(size_t j = 0; j < ncentres; ++j) {
			// % Pick out data points belonging to this centre
			MatrixXf tmp = post.col(j);
			size_t nonZero = 0;
			for(size_t kk = 0; kk < (size_t)tmp.rows(); ++kk) {
				nonZero += (tmp(kk, 0) != 0);
			}
			MatrixXf c = MatrixXf::Zero(nonZero, x.cols());

			size_t idx = 0;
			for(size_t t = 0; t < (size_t)tmp.rows(); ++t) {
				if(tmp(t,0)) {
					c.row(idx) = x.row(t);
					++idx;
				}
			}
			MatrixXf diffs = c.array() - (MatrixXf::Ones(c.rows(), 1) * Centres.row(j)).array(); //diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));
			covars(0,0).row(j) = (diffs.cwiseProduct(diffs)).colwise().sum(); // mix.covars(j, :) = sum((diffs.*diffs), 1)/size(c, 1);
			covars(0,0).row(j) /= c.rows();
			// % Replace small entries by GMM_WIDTH value
			tmp = (covars(0,0).row(j).array() < std::numeric_limits<double>::epsilon( )).cast<float>();
			covars(0,0).row(j) += gmm_width * ( tmp ); // mix.covars(j, :) = mix.covars(j, :) + GMM_WIDTH.*(mix.covars(j, :)<eps);
		}
	}
	else { // full
		for(size_t j = 0; j < ncentres; ++j) {
			// % Pick out data points belonging to this centre
			MatrixXf tmp = post.col(j);
			size_t nonZero = 0;
			for(size_t kk = 0; kk < (size_t)tmp.rows(); ++kk) {
				nonZero += (tmp(kk, 0) != 0);
			}
			MatrixXf c = MatrixXf::Zero(nonZero, x.cols());

			size_t idx = 0;
			for(size_t t = 0; t < (size_t)tmp.rows(); ++t) {
				if(tmp(t,0)) {
					c.row(idx) = x.row(t);
					++idx;
				}
			}

			MatrixXf diffs = c.array() - (MatrixXf::Ones(c.rows(), 1) * Centres.row(j)).array(); //diffs = c - (ones(size(c, 1), 1) * mix.centres(j, :));

			covars(j, 0) = diffs.adjoint() * diffs; // mix.covars(j, :) = sum((diffs.*diffs), 1)/size(c, 1);
			covars(j, 0) /= c.rows();
			// % Add GMM_WIDTH*Identity to rank-deficient covariance matrices

#ifdef DEBUG_GMM
		std::cout << "covars = " << std::endl;
		print4DMatrix(covars);
#endif

			if ( rank(covars(j,0)) < dim) {
				covars(j,0) += gmm_width * MatrixXf::Identity(dim, dim);
			}
		}
	}
#ifdef DEBUG_GMM
		std::cout << "covars = " << std::endl;
		print4DMatrix(covars);
#endif
	Covars = covars;

}
const MatrixXf& GMM::getCentres() const { return Centres;}
const MultiD& GMM::getCovars() const { return Covars;}

} /* namespace HMM */
