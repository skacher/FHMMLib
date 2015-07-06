/*
 * KMeansInit.cpp
 *
 *  Created on: 13 мая 2015 г.
 *      Author: kacher
 */

#include <stdexcept>
#include <vector>

#include "../../base/KMeansInit.h"
#include "../../base/MatlabUtils.h"

#include "../../io/Util.h"

#ifdef DEBUG_KMEANS_INIT
#include <iostream>
#endif

namespace HMM {

#ifdef DEBUG_GMM
KMeansInit::KMeansInit(
		const MatrixXf& centres,
		const MatrixXf& data,
		const std::string& filename
) : Centres(centres)
#else
KMeansInit::KMeansInit(
		const MatrixXf& centres,
		const MatrixXf& data
) : Centres(centres)
#endif
{

	size_t ndata = data.rows();
	size_t data_dim =  data.cols(); //[ndata, data_dim] = size(data);

	size_t ncentres = centres.rows();
	size_t dim = centres.cols(); //	[ncentres, dim] = size(centres);

	if(dim != data_dim)
		throw  std::runtime_error("KMeansInit: Data dimension does not match dimension of centres");

	if (ncentres > ndata)
		throw  std::runtime_error("KMeansInit: More centres than data");

	size_t niters = 5;


#ifdef DEBUG_GMM
	MatrixXf t = Read2DMatrix(filename);
	std::vector<size_t> perm;
	for(size_t kk = 0; kk < (size_t)t.cols(); ++kk) {
		perm.push_back((size_t)(t(0, kk) - 1));
	}
#else
	std::vector<size_t> perm = randperm(ndata);
#endif


	for(size_t center = 0; center < ncentres; ++center) {
		Centres.row(center) = data.row(perm[center]);
	}

	//% Matrix to make unit vectors easy to construct
	MatrixXf id = MatrixXf::Identity(ncentres, ncentres); //id = eye(ncentres);

	MatrixXf old_centres;
	double old_e = 0.0;
	for(size_t n = 0; n < niters; ++n) {
		old_centres = Centres; //% Save old centres to check for termination

		 //% Calculate posteriors based on existing centres
		 MatrixXf d2 = dist2(data, Centres);

		 MatrixXf minvals;
		 MatrixXf index;
		 min_with_index(d2.adjoint(), 1, minvals, index); // [minvals, index] = min(d2', [], 1);

#ifdef DEBUG_KMEANS_INIT
		 std::cout << "d2 = " << std::endl << d2 << std::endl;
		 std::cout << "minvals = " << std::endl << minvals << std::endl;
		 std::cout << "index = " << std::endl << index << std::endl;
#endif

		 Post = MatrixXf::Zero(ndata,ncentres);

		 for(size_t i = 0; i < (size_t)index.rows(); ++i) {
			 Post.row(i) = id.row(index(i,0));
		 }//post = id(index,:);

#ifdef DEBUG_KMEANS_INIT
		 std::cout << "Post = " << std::endl << Post << std::endl;
#endif

		 MatrixXf num_points = Post.colwise().sum(); //num_points = sum(post, 1);
#ifdef DEBUG_KMEANS_INIT
		 std::cout << "num_points = " << std::endl << num_points << std::endl;
#endif
		 for(size_t j = 0; j < ncentres; ++j) {
			 if(num_points(0, j) > 0) { //centres(j,:) = sum(data(find(post(:,j)),:), 1)/num_points(j);
				 MatrixXf tmp = Post.col(j);
				 size_t nonZero = 0;
				 for(size_t kk = 0; kk < (size_t)tmp.rows(); ++kk) {
					 nonZero += (tmp(kk, 0) != 0);
				 }
				 MatrixXf tmpData = MatrixXf::Zero(nonZero, data_dim);

				 size_t idx = 0;
				 for(size_t t = 0; t < (size_t)tmp.rows(); ++t) {
					 if(tmp(t,0)) {
						 tmpData.row(idx) = data.row(t);
						 ++idx;
					 }
				 }
				 Centres.row(j) = tmpData.colwise().sum() / num_points(0,j);
			 }
		 }

#ifdef DEBUG_KMEANS_INIT
		 std::cout << "Centres = " << std::endl << Centres << std::endl;
#endif
		 // % Error value is total squared distance from cluster centres
		 double e = (minvals.colwise().sum())(0,0);

		 if (n > 0) {
			 	 MatrixXf d = (Centres - old_centres).cwiseAbs();
			 	 MatrixXf _max_value;
			 	 MatrixXf _max_index;
			 	 MatrixXf result_value;
			 	 MatrixXf result_index;

			 	 max_with_index(d, 1, _max_value, _max_index); //max along column
			 	 max_with_index(_max_value, 2, result_value, result_index); //max along row


			 	 if ( (fabs(old_e - e) < 0.0001) && (result_value(0,0) < 0.0001) ){ // max(max(abs(centres - old_centres))) < options(2) &  abs(old_e - e) < options(3)
			 		 return;
			 	 }
		 }
		 old_e = e;

	}

}

const MatrixXf& KMeansInit::getCentres() const { return Centres; }
const MatrixXf& KMeansInit::getPost() const { return Post; }

} /* namespace HMM */
