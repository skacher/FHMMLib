/*
 * MixGaussProb.h
 *
 *  Created on: 22 апр. 2015 г.
 *      Author: kacher
 */

#ifndef MIXGAUSSPROB_H_
#define MIXGAUSSPROB_H_

#include "HMMTypes.h"

namespace HMM {

class MixGaussProb {
public:
	MixGaussProb();
/*
 * % EVAL_PDF_COND_MOG Evaluate the pdf of a conditional mixture of Gaussians
% function [B, B2] = eval_pdf_cond_mog(data, mu, Sigma, mixmat, unit_norm)
%
% Notation: Y is observation, M is mixture component, and both may be conditioned on Q.
% If Q does not exist, ignore references to Q=j below.
% Alternatively, you may ignore M if this is a conditional Gaussian.
%
% INPUTS:
% data(:,t) = t'th observation vector
%
% mu(:,k) = E[Y(t) | M(t)=k]
% or mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k]
%
% Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
% or there are various faster, special cases:
%   Sigma() - scalar, spherical covariance independent of M,Q.
%   Sigma(:,:) diag or full, tied params independent of M,Q.
%   Sigma(:,:,j) tied params independent of M.
%
% mixmat(k) = Pr(M(t)=k) = prior
% or mixmat(j,k) = Pr(M(t)=k | Q(t)=j)
% Not needed if M is not defined.
%
% unit_norm - optional; if 1, means data(:,i) AND mu(:,i) each have unit norm (slightly faster)
%
% OUTPUT:
% B(t) = Pr(y(t))
% or
% B(i,t) = Pr(y(t) | Q(t)=i)
% B2(i,k,t) = Pr(y(t) | Q(t)=i, M(t)=k)
%
% If the number of mixture components differs depending on Q, just set the trailing
% entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
% then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.
 *
 */
	MixGaussProb(
			const MatrixXf& data,
			const MultiD& mu,
			const MultiD& Sigma,
			const MatrixXf& mixmat
	);
	const MatrixXf& getOutputPDFValueConditionedMC() const;
	const MultiD& getOutputPDFValueConditionedMCandGMM() const;
private:
	MatrixXf B;
	MultiD B2;
};

} /* namespace HMM */

#endif /* MIXGAUSSPROB_H_ */
