/*
 * FwdBack.h
 *
 *  Created on: 22 апр. 2015 г.
 *      Author: kacher
 */

#ifndef FWDBACK_H_
#define FWDBACK_H_

#include "HMMTypes.h"

namespace HMM {

class FwdBack {
public:
	FwdBack();

	/*
	 * FWDBACK Compute the posterior probs. in an HMM using the forwards backwards algo.
%
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(init_state_distrib, transmat, obslik, ...)
%
% Notation:
% Y(t) = observation, Q(t) = hidden state, M(t) = mixture variable (for MOG outputs)
% A(t) = discrete input (action) (for POMDP models)
%
% INPUT:
% init_state_distrib(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
%  or transmat{a}(i,j) = Pr(Q(t) = j | Q(t-1)=i, A(t-1)=a) if there are discrete inputs
% obslik(i,t) = Pr(Y(t)| Q(t)=i)
%   (Compute obslik using eval_pdf_xxx on your data sequence first.)
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% For HMMs with MOG outputs: if you want to compute gamma2, you must specify
% 'obslik2' - obslik(i,j,t) = Pr(Y(t)| Q(t)=i,M(t)=j)  []
% 'mixmat' - mixmat(i,j) = Pr(M(t) = j | Q(t)=i)  []
%  or mixmat{t}(m,q) if not stationary
%
% For HMMs with discrete inputs:
% 'act' - act(t) = action performed at step t
%
% Optional arguments:
% 'fwd_only' - if 1, only do a forwards pass and set beta=[], gamma2=[]  [0]
% 'scaled' - if 1,  normalize alphas and betas to prevent underflow [1]
% 'maximize' - if 1, use max-product instead of sum-product [0]
%
% OUTPUTS:
% alpha(i,t) = p(Q(t)=i | y(1:t)) (or p(Q(t)=i, y(1:t)) if scaled=0)
% beta(i,t) = p(y(t+1:T) | Q(t)=i)*p(y(t+1:T)|y(1:t)) (or p(y(t+1:T) | Q(t)=i) if scaled=0)
% gamma(i,t) = p(Q(t)=i | y(1:T))
% loglik = log p(y(1:T))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:T))  - NO LONGER COMPUTED
% xi_summed(i,j) = sum_{t=}^{T-1} xi(i,j,t)  - changed made by Herbert Jaeger
% gamma2(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)
%
% If fwd_only = 1, these become
% alpha(i,t) = p(Q(t)=i | y(1:t))
% beta = []
% gamma(i,t) = p(Q(t)=i | y(1:t))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:t))
% gamma2 = []
%
% Note: we only compute xi if it is requested as a return argument, since it can be very large.
% Similarly, we only compute gamma2 on request (and if using MOG outputs).
%
% Examples:
%
% [alpha, beta, gamma, loglik] = fwdback(pi, A, multinomial_prob(sequence, B));
%
% [B, B2] = mixgauss_prob(data, mu, Sigma, mixmat);
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(pi, A, B, 'obslik2', B2, 'mixmat', mixmat);
	 *
	 */
	FwdBack(
			const MatrixXf& prior,
			const MatrixXf& transmat,
			const MatrixXf& B,
			const MultiD& B2,
			const MatrixXf& mixmat,
			size_t out_args = 5
	);
	const MatrixXf& getAlpha() const;
	const MatrixXf& getGamma() const;
	const MatrixXf& getBeta() const;
	double getLogLik() const;
	const MatrixXf& getXiSummed() const;
	const MultiD& getGamma2() const;
private:
	MatrixXf Alpha;
	MatrixXf Gamma;
	MatrixXf Beta;
	double LogLikelihood;
	MatrixXf XiSummed;
	MultiD Gamma2;
};

} /* namespace HMM */

#endif /* FWDBACK_H_ */
