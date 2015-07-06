/*
 * MixgaussMstep.cpp
 *
 *  Created on: 23 апр. 2015 г.
 *      Author: kacher
 */

#include <iostream>

#include "../../base/MixgaussMstep.h"
#include "../../base/MatlabUtils.h"

#include "../../io/Util.h"
#include "../../base/Config.h"

namespace HMM {

MixgaussMstep::MixgaussMstep() {

}
MixgaussMstep::MixgaussMstep(
		const MatrixXf _w,
		const MultiD& Y,
		const MultiD& _YY,
		const MatrixXf& YTY,
		const std::string& cov_type
	) {
#ifdef DEBUG_MIXGAUSS_MSTEP
	/*
	std::cout << cov_type << std::endl;
	std::cout << "_w = " << std::endl << _w << std::endl;

	std::cout << "Y = " << std::endl;
	print4DMatrix(Y);

	std::cout << "YY = " << std::endl;
	print4DMatrix(_YY);

	std::cout << "YTY = " << std::endl << YTY << std::endl;*/
#endif

	MatrixXf w = _w;
	MultiD YY = _YY;

	size_t Ysz = Y(0,0).rows(); //check if Y not empty
	size_t Q = Y.size() * Y(0,0).cols();//[Ysz Q] = size(Y);
	MatrixXf N = w.colwise().sum(); //N = sum(w);
	MatrixXf cov_prior_tmp = 0.01 * MatrixXf::Identity(Ysz,Ysz);
	MultiD cov_prior = repmat<MatrixXf, MultiD>(cov_prior_tmp, 1,1,Q,1);//cov_prior = repmat(0.01*eye(Ysz,Ysz), [1 1 Q]);

	YY = reshape4Dto4D(YY, Ysz, Ysz, Q, 1); //YY = reshape(YY, [Ysz Ysz Q]);
	MatrixXf eq_zero = (w.array() == MatrixXf::Zero(w.rows(), w.cols()).array()).cast<float>();
	w = w + eq_zero;//w = w + (w==0);

	mu = MatrixXf::Zero(Ysz, Q);
	for(size_t i = 0; i < Q; ++i) {
		double wi = w(i % w.rows(), int(i / w.rows()));
		size_t gr = size_t(i / Y(0,0).cols());
		size_t lc = i % Y(0,0).cols();
		mu.col(i) = Y(gr,0).col(lc) / wi; //!!! // mu(:,i) = Y(:,i) / w(i);
	}

	Sigma = MultiD(Q, 1); //Sigma = zeros(Ysz,Ysz,Q);
	for(size_t i = 0; i < Q; ++i) {
		Sigma(i, 0) = MatrixXf::Zero(Ysz, Ysz);
	}

	for(size_t i = 0; i < Q; ++i) {
		double wi = w(i % w.rows(), int(i / w.rows()));
		if(cov_type == "spherical") {
			size_t r = i % YTY.rows();
			size_t c = int(i / YTY.rows());
			double s = (1.0 / Ysz) * ( (YTY(r, c) / wi)  - (mu.col(i).adjoint() * mu.col(i))); //s2 = (1/Ysz)*( (YTY(i)/w(i)) - mu(:,i)'*mu(:,i) );
			Sigma(i,0) = s * MatrixXf::Identity(Ysz,Ysz); //Sigma(:,:,i) = s2 * eye(Ysz);
		} else {
			MatrixXf SS = (YY(i, 0) / wi )  - (mu.col(i) * mu.col(i).adjoint()); //SS = YY(:,:,i)/w(i)  - mu(:,i)*mu(:,i)';
			if (cov_type == "diag") {
				MatrixXf diagonal = SS.diagonal();
				SS = diagonal.asDiagonal(); //SS = diag(diag(SS));
			}
			Sigma(i,0) = SS; //Sigma(:,:,i) = SS;
		}
	}
	for(size_t r = 0; r < (size_t)Sigma.rows(); ++r) {
		for(size_t c = 0; c < (size_t)Sigma.cols(); ++c) {
			Sigma(r,c) += cov_prior(r,c);
		}
	}
#ifdef DEBUG_MIXGAUSS_MSTEP
	std::cout << "mu = " << std::endl << mu << std::endl;
	print4DMatrix(Sigma);
#endif
}

} /* namespace HMM */
