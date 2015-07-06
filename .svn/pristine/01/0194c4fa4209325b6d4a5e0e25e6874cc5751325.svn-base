/*
 * MHMMEM.h
 *
 *  Created on: 21 апр. 2015 г.
 *      Author: kacher
 */

#ifndef MHMMEM_H_
#define MHMMEM_H_

#include <vector>
#include "HMMTypes.h"
#include "ESSMHMM.h"

namespace HMM {

class MHMM_EM_Params {

public:
	MHMM_EM_Params();
	void setMaxIter(size_t max_iter);
	void setThreshold(double max_iter);
	void setCovType(const std::string& cov_type);

	void setAdjPrior(size_t adj_prior);
	void setAdjTrans(size_t adj_trans);
	void setAdjMix(size_t adj_mix);
	void setAdjMu(size_t adj_mu);
	void setAdjSigma(size_t adj_sigma);


	/*
	void print() const {
		std::cout << "MaxIter = " <<  MaxIter << std::endl;
		std::cout << "Threshold = " <<  Threshold << std::endl;
		std::cout << "CovType = " <<  CovType << std::endl;
		std::cout << "AdjPrior = " <<  AdjPrior << std::endl;
		std::cout << "AdjTrans = " <<  AdjTrans << std::endl;
		std::cout << "AdjMix = " <<  AdjMix << std::endl;

		std::cout << "AdjMu = " <<  AdjMu << std::endl;
		std::cout << "AdjSigma = " <<  AdjSigma << std::endl;
	}*/
private:
	size_t MaxIter;
	double Threshold;
	std::string CovType;
	size_t AdjPrior;
	size_t AdjTrans;
	size_t AdjMix;
	size_t AdjMu;
	size_t AdjSigma;

	friend class MHMM_EM;
};


class MHMM_EM {
public:
	MHMM_EM();
	/*% LEARN_MHMM Compute the ML parameters of an HMM with (mixtures of) Gaussians output using EM.
% [ll_trace, prior, transmat, mu, sigma, mixmat] = learn_mhmm(data, ...
%   prior0, transmat0, mu0, sigma0, mixmat0, ...)
%
% Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
%
% INPUTS:
% data{ex}(:,t) or data(:,t,ex) if all sequences have the same length
% prior(i) = Pr(Q(1) = i),
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k ]
% Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
% mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to [] or ones(Q,1) if only one mixture component
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% 'max_iter' - max number of EM iterations [10]
% 'thresh' - convergence threshold [1e-4]
% 'verbose' - if 1, print out loglik at every iteration [1]
% 'cov_type' - 'full', 'diag' or 'spherical' ['full']
%
% To clamp some of the parameters, so learning does not change them:
% 'adj_prior' - if 0, do not change prior [1]
% 'adj_trans' - if 0, do not change transmat [1]
% 'adj_mix' - if 0, do not change mixmat [1]
% 'adj_mu' - if 0, do not change mu [1]
% 'adj_Sigma' - if 0, do not change Sigma [1]
%
% If the number of mixture components differs depending on Q, just set  the trailing
% entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
% then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.
%
% OUTPUTS:
% LL - log-likelihood value
% prior - a prior distribution of HMM
% transmat - a transition probabilities matrix of HMM
% mu, Sigma - mean vectors and covariance matrices of MOG components
% mixmat - a matrix of MOG mixture probabilites for each of for each output

% posterior - a structure with two cells (it contains only one cell if the output)
% is modelled by a Gaussian distribution. The first cell contains posterior
% probabilities p(Q(t)=i | y(1:T)) of HMM states. The second state contains
% posteior probabilities(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)
	 *
	*/
	MHMM_EM(
			const MatrixXf& data,
			const MatrixXf& prior,
			const MatrixXf& transmat,
			const MultiD& mu,
			const MultiD& Sigma,
			const MatrixXf& mixmat,
			const MHMM_EM_Params& params);


	const std::vector<double>& getLogLikelihood() const;
	const MatrixXf& getPrior() const;
	const MatrixXf& getTransmat() const;
	const MultiD& getMu() const;
	const MultiD& getSigma() const;
	const MatrixXf& getMixmat() const;


	const MatrixXf& PostProbMC() const; //Gamma
	const MultiD& PostProbMCandMOG() const; //Gamma2

private:
	/*% EM_CONVERGED Has EM converged?

% [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
%
% EM has converged if the slope of the log-likelihood function falls below 'threshold',
% i.e., |f(t) - f(t-1)| / avg < threshold,
% where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
% 'threshold' defaults to 1e-4.
%
% This stopping criterion is from Numerical Recipes in C p423
%
% If we are doing MAP estimation (using priors), the likelihood can decrease,
% even though the mode of the posterior is increasing.
*/
	bool EmConverged(double loglik, double previous_loglik) const;

	void CheckMatrixSize(const MatrixXf& data);

private:
	std::vector<double> LogLikelihood;
	MatrixXf Prior;
	MatrixXf Transmat;
	MultiD Mu;
	MultiD Sigma;
	MatrixXf Mixmat;
	MHMM_EM_Params Params;

	ESS_MHMM ess_mhmm;

};


/*
function [LL, prior, transmat, mu, Sigma, mixmat, posterior] =  mhmm_em(data, prior, transmat, mu, Sigma, mixmat, varargin)
[LL, prior1, transmat1, mu1, Sigma1, mixmat1, posterior1] = mhmm_em(obs, prior0, transmat0, mu0, Sigma0, mixmat0, 'max_iter', 10);
*/

} /* namespace HMM */

#endif /* MHMMEM_H_ */
