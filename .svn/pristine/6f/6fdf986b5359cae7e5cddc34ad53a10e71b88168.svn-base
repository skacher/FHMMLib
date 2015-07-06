/*
 * MixGaussInit.h
 *
 *  Created on: 11 мая 2015 г.
 *      Author: kacher
 */

#ifndef MIXGAUSSINIT_H_
#define MIXGAUSSINIT_H_

#include "HMMTypes.h"
#include "Config.h"

namespace HMM {

class MixGaussInit {
public:
	MixGaussInit();
	/*
	 * % MIXGAUSS_INIT Initial parameter estimates for a mixture of Gaussians
% function [mu, Sigma, weights] = mixgauss_init(M, data, cov_type. method)
%
% INPUTS:
% data(:,t) is the t'th example
% M = num. mixture components
% cov_type = 'full', 'diag' or 'spherical'
% method = 'rnd' (choose centers randomly from data) or 'kmeans' (needs netlab)
%
% OUTPUTS:
% mu(:,k)
% Sigma(:,:,k)
% weights(k)
	 */
#ifdef DEBUG_GMM
	MixGaussInit(size_t _q , size_t _m, size_t _o, const MatrixXf& obs, const std::string& cov_type, const std::string& init_type, const std::string& filename);
#else
	MixGaussInit(size_t _q , size_t _m, size_t _o, const MatrixXf& obs, const std::string& cov_type, const std::string& init_type);
#endif
	const MultiD& getSigma() const;
	const MultiD& getMu() const;
private:
	MultiD Sigma;
	MatrixXf Mu; //Mu to transform
	MultiD RealMu;
};

} /* namespace HMM */

#endif /* MIXGAUSSINIT_H_ */
