/*
 * MixGaussProb.cpp
 *
 *  Created on: 22 апр. 2015 г.
 *      Author: kacher
 */

#include <iostream>
#include <stdexcept>

#include "../../base/MixGaussProb.h"
#include "../../base/MatlabUtils.h"
#include "../../io/Util.h"

#include "../../base/Config.h"

namespace HMM {

MixGaussProb::MixGaussProb() {
}

MixGaussProb::MixGaussProb(
		const MatrixXf& data,
		const MultiD& mu,
		const MultiD& Sigma,
		const MatrixXf& _mixmat
) {
	MatrixXf mixmat = _mixmat;
	/*
	if iscolumn(mu)
	  d = length(mu);
	  Q = 1; M = 1;
	elseif ismatrix(mu)
	  [~, Q] = size(mu);
	  M = 1;
	else
	  [~, Q M] = size(mu);
	end*/
	//we always have 3D mu

	size_t Q = mu(0,0).cols(); // not safe
	size_t M = mu.rows();

	size_t T = data.cols(); //[d T] = size(data);

	if(mixmat.rows() == 0 && mixmat.cols() == 0) { // if nargin < 4, mixmat = ones(Q,1); end
		mixmat = MatrixXf::Ones(Q, 1);
	}

	//sigma always 4d, so implement % general case
	B2.resize(T,1); //B2 = zeros(Q,M,T);
	for(size_t i = 0; i < T; ++i) {
		B2(i, 0) = MatrixXf::Zero(Q, M);
	}

#ifdef DEBUG_GAUSSIAN_PROB
	print4DMatrix(mu);
#endif

	for(size_t j = 0; j < Q; ++j) { //for j=1:Q
		for(size_t k = 0; k < M; ++k) { //for k=1:M
			MatrixXf sig = Sigma(j, k);
			MatrixXf m = mu(k, 0).col(j);
			MatrixXf t = gaussianProb(data, m, sig);
			for(size_t index = 0; index < (size_t)t.rows(); ++index) {
				B2(index, 0)(j, k) = t(index, 0); //B2(j,k,:) = gaussian_prob(data, mu(:,j,k), Sigma(:,:,j,k));
			}
		}
	}
	B = MatrixXf::Zero(Q,T);//B = zeros(Q,T);

	if(Q < T) {
		for(size_t q = 0; q < Q; ++q) {
			MatrixXf tmp(M, T);
			for(size_t i = 0; i < T; ++i) {
				MatrixXf B2i1 = B2(i,0);
				tmp.col(i) = B2(i,0).row(q);
			} //permute(B2(q,:,:), [2 3 1])
			B.row(q) = mixmat.row(q) * tmp;
			// B(q,:) = mixmat(q,:) * permute(B2(q,:,:), [2 3 1]); % vector * matrix sums over m
		}
	}
	else {
		std::string unimplemented_code = "for t=1:T\nB(:,t) = sum(mixmat .* B2(:,:,t), 2); % sum over m\nend\n";
		throw  std::runtime_error(unimplemented_code);
	}
#ifdef DEBUG_MIX_GAUSS_PROB
	std::cout << B << std::endl;
	print4DMatrix(B2);
#endif
}

const MatrixXf& MixGaussProb::getOutputPDFValueConditionedMC() const {return B;}
const MultiD& MixGaussProb::getOutputPDFValueConditionedMCandGMM() const {return B2;}

} /* namespace HMM */
