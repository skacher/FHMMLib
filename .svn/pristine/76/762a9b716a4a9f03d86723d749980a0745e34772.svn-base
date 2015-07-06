/*
 * MixGaussInit.cpp
 *
 *  Created on: 11 мая 2015 г.
 *      Author: kacher
 */

#include <iostream>
#include <stdexcept>
#include <time.h>

#include "../../base/MixGaussInit.h"
#include "../../base/GMM.h"
#include "../../io/Util.h"
#include "../../base/MatlabUtils.h"
#include "../../base/Config.h"


namespace HMM {


MixGaussInit::MixGaussInit() {
	// TODO Auto-generated constructor stub

}

#ifdef DEBUG_GMM
MixGaussInit::MixGaussInit(size_t _q , size_t _m, size_t _o, const MatrixXf& data, const std::string& covar_type, const std::string& mode, const std::string& filename) {
#else
MixGaussInit::MixGaussInit(size_t _q , size_t _m, size_t _o, const MatrixXf& data, const std::string& covar_type, const std::string& mode) {
#endif
	size_t M = _q * _m;
	size_t d = data.rows();
	size_t T = data.cols();
	srand (time(NULL));

	if (mode == "rnd") {
		Mu = MatrixXf::Zero(data.rows(), M);

		MatrixXf C = cov<MatrixXf>(data.adjoint());

		MatrixXf diagonal = C.diagonal();
		C = diagonal.asDiagonal(); //SS = diag(diag(c));
		C = C * 0.5;

		Sigma = repmat<MatrixXf, MultiD>(C, 1, 1, M, 1);


	#ifdef DEBUG_MIXGAUSS_INIT
		MatrixXf t = Read2DMatrix(filename);
		std::vector<size_t> indices;
		for(size_t kk = 0; kk < (size_t)t.cols(); ++kk) {
			indices.push_back((size_t)(t(0, kk) - 1));
		}
	#else
		std::vector<size_t> indices = randperm(T);
	#endif

		for(size_t i = 0; i < M; ++i) {
			Mu.col(i) = data.col(indices[i]);
		}



	}
	else if (mode == "kmeans") {
		#ifdef DEBUG_GMM
			GMM gmm(d, M, data.adjoint(), covar_type, filename);
		#else
			GMM gmm(d, M, data.adjoint(), covar_type);
		#endif

			Mu = gmm.getCentres().adjoint();

			MultiD covar = gmm.getCovars();

			Sigma = MultiD(M, 1);

		#ifdef DEBUG_GMM
			std::cout << "Mu = " << std::endl << Mu << std::endl;
			std::cout << "covar = " << std::endl;
			print4DMatrix(covar);
		#endif

			for (size_t m = 0; m < M; ++m) {
				if (covar_type == "spherical") {
					Sigma(m,0) = covar(0,0)(m,0) * MatrixXf::Identity(d,d);
				}
				else if(covar_type == "diag") {
					Sigma(m,0) = covar(0,0).row(m).adjoint().asDiagonal();
				}
				else { // full
					Sigma(m, 0) = covar(m,0);
				}
			}
	}
	else {
		throw  std::runtime_error("mode is unknown, it should be 'rnd' or 'kmeans' ");
	}

#ifdef DEBUG_GMM
	std::cout << "Mu = " << std::endl << Mu << std::endl;
	std::cout << "Sigma = " << std::endl;
	print4DMatrix(Sigma);
#endif
	RealMu = reshape2Dto4D(Mu, _o, _q, _m, 1);
	Sigma = reshape4Dto4D(Sigma, _o, _o, _q, _m);
}

const MultiD& MixGaussInit::getSigma() const {return Sigma;}
const MultiD& MixGaussInit::getMu() const {return RealMu;}

} /* namespace HMM */
