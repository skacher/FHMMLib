/*
 * MixgaussMstep.h
 *
 *  Created on: 23 апр. 2015 г.
 *      Author: kacher
 */

#ifndef MIXGAUSSMSTEP_H_
#define MIXGAUSSMSTEP_H_


#include "HMMTypes.h"

namespace HMM {

class MixgaussMstep {
public:
	MixgaussMstep();
	/*% MSTEP_COND_GAUSS Compute MLEs for mixture of Gaussians given expected sufficient statistics
% function [mu, Sigma] = Mstep_cond_gauss(w, Y, YY, YTY, varargin)
%
% We assume P(Y|Q=i) = N(Y; mu_i, Sigma_i)
% and w(i,t) = p(Q(t)=i|y(t)) = posterior responsibility
% See www.ai.mit.edu/~murphyk/Papers/learncg.pdf.
%
% INPUTS:
% w(i) = sum_t w(i,t) = responsibilities for each mixture component
%  If there is only one mixture component (i.e., Q does not exist),
%  then w(i) = N = nsamples,  and
%  all references to i can be replaced by 1.
% YY(:,:,i) = sum_t w(i,t) y(:,t) y(:,t)' = weighted outer product
% Y(:,i) = sum_t w(i,t) y(:,t) = weighted observations
% YTY(i) = sum_t w(i,t) y(:,t)' y(:,t) = weighted inner product
%   You only need to pass in YTY if Sigma is to be estimated as spherical.
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% 'cov_type' - 'full', 'diag' or 'spherical' ['full']
% 'tied_cov' - 1 (Sigma) or 0 (Sigma_i) [0]
% 'clamped_cov' - pass in clamped value, or [] if unclamped [ [] ]
% 'clamped_mean' - pass in clamped value, or [] if unclamped [ [] ]
% 'cov_prior' - Lambda_i, added to YY(:,:,i) [0.01*eye(d,d,Q)]
%
% If covariance is tied, Sigma has size d*d.
% But diagonal and spherical covariances are represented in full size.
	 */
	MixgaussMstep(
		const MatrixXf w,
		const MultiD& Y,
		const MultiD& YY,
		const MatrixXf& YTY,
		const std::string& cov_type
	);
	const MultiD& getSigma() const {return Sigma;}
	const MatrixXf& getMu() const {return mu;}

private:
	MultiD Sigma;
	MatrixXf mu;
};


} /* namespace HMM */

#endif /* MIXGAUSSMSTEP_H_ */
