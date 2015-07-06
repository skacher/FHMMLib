/*
 * ESSMHMM.h
 *
 *  Created on: 22 апр. 2015 г.
 *      Author: kacher
 */

#ifndef ESSMHMM_H_
#define ESSMHMM_H_

#include "HMMTypes.h"
#include "FwdBack.h"

namespace HMM {

class ESS_MHMM {
public:
	ESS_MHMM();
	/*
	% ESS_MHMM Compute the Expected Sufficient Statistics for a MOG Hidden Markov Model.
	%
	% Outputs:
	% exp_num_trans(i,j)   = sum_l sum_{t=2}^T Pr(Q(t-1) = i, Q(t) = j| Obs(l))
	% exp_num_visits1(i)   = sum_l Pr(Q(1)=i | Obs(l))
	%
	% Let w(i,k,t,l) = P(Q(t)=i, M(t)=k | Obs(l))
	% where Obs(l) = Obs(:,:,l) = O_1 .. O_T for sequence l
	% Then
	% postmix(i,k) = sum_l sum_t w(i,k,t,l) (posterior mixing weights/ responsibilities)
	% m(:,i,k)   = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)
	% ip(i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)' * Obs(:,t,l)
	% op(:,:,i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l) * Obs(:,t,l)'
	 */
	ESS_MHMM(
			const MatrixXf& prior,
			const MatrixXf& transmat,
			const MatrixXf& mixmat,
			const MultiD& mu,
			const MultiD& Sigma,
			const MatrixXf& data
	);

	double getLogLikelihood();
	const MatrixXf& getPostmix() const;
	const MatrixXf& getExpNumTrans() const;
	const MatrixXf& getExpNumVisits1() const;
	const MultiD& getM() const;
	const MatrixXf& getIp() const;
	const MultiD& getOp() const;

	const MatrixXf& getGamma() const;
	const MultiD& getGamma2() const;


private:
	double LogLikelihood;
	MatrixXf Postmix;
	MatrixXf ExpNumTrans;
	MatrixXf ExpNumVisits1;
	MultiD m;
	MatrixXf Ip;
	MultiD Op;
	MatrixXf Gamma;
	MultiD Gamma2;

	FwdBack fb;
};

} /* namespace HMM */

#endif /* ESSMHMM_H_ */
