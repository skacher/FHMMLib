/*
 * ESSMHMM.cpp
 *
 *  Created on: 22 апр. 2015 г.
 *      Author: kacher
 */

#include <iostream>

#include "../../base/ESSMHMM.h"
#include "../../base/MatlabUtils.h"
#include "../../base/MixGaussProb.h"
#include "../../io/Util.h"
#include "../../base/Config.h"

namespace HMM {

ESS_MHMM::ESS_MHMM() : LogLikelihood(0.0)  {
}

ESS_MHMM::ESS_MHMM(
		const MatrixXf& prior,
		const MatrixXf& transmat,
		const MatrixXf& mixmat,
		const MultiD& mu,
		const MultiD& Sigma,
		const MatrixXf& data
) : LogLikelihood(0.0) {
	//in our case we don't use cell as input. We always use uncell data
	size_t O = data.rows(); //O = size(data{1},1);
	size_t Q = prior.size(); // Q = length(prior);
	size_t M = mixmat.cols(); //M = size(mixmat,2);

	ExpNumTrans =  MatrixXf::Zero(Q,Q); //exp_num_trans = zeros(Q,Q);
	ExpNumVisits1 = MatrixXf::Zero(Q, 1);// exp_num_visits1 = zeros(Q,1);
	Postmix = MatrixXf::Zero(Q, M);//postmix = zeros(Q,M);

	m = MultiD(M, 1); //m = zeros(O,Q,M);
	for(size_t i = 0; i < M; ++i) {
		m(i, 0) = MatrixXf::Zero(O, Q);
	}
	Op = MultiD(Q, M);  //op = zeros(O,O,Q,M);
	for(size_t i = 0; i < Q; ++i) {
		for(size_t j = 0; j < M; ++j) {
			Op(i, j) = MatrixXf::Zero(O, O);
		}
	}

	Ip = MatrixXf::Zero(Q,M); //ip = zeros(Q,M);
	bool mix = (M > 1);

	MatrixXf obs = data; //optimize it!
	size_t T = obs.cols(); // T = size(obs,2)


	if (mix) {
		MixGaussProb mgp(obs, mu, Sigma, mixmat);
		fb = FwdBack(prior, transmat, mgp.getOutputPDFValueConditionedMC(), mgp.getOutputPDFValueConditionedMCandGMM(), mixmat, 6);
		Gamma2 = fb.getGamma2();
	}
	else {
		MatrixXf tmp;
		MixGaussProb mgp(obs, mu, Sigma, tmp);
		fb = FwdBack(prior, transmat, mgp.getOutputPDFValueConditionedMC(), MultiD(), MatrixXf(), 6);
	}

	Gamma = fb.getGamma();

	LogLikelihood += fb.getLogLik(); //loglik = loglik +  current_loglik;
	ExpNumTrans += fb.getXiSummed(); //exp_num_trans = exp_num_trans + xi_summed; % sum(xi,3);
	ExpNumVisits1 += Gamma.col(0); //exp_num_visits1 = exp_num_visits1 + gamma(:,1);


	if(mix) {
		MatrixXf sum3 = MatrixXf::Zero(Q, M); //postmix = MatrixXf::Zero(Q, M);//postmix = zeros(Q,M);
		for(size_t i = 0; i < (size_t)Gamma2.rows(); ++i) {
			sum3 += Gamma2(i, 0);
		}
		Postmix += sum3;
	}
	else {
		Postmix += Gamma.rowwise().sum();//postmix = postmix + sum(gamma,2);
		Gamma2 = reshape2Dto4D(Gamma, Q, 1, T, 1);//gamma2 = reshape(gamma, [Q 1 T]); % gamma2(i,m,t) = gamma(i,t)
	}

#ifdef DEBUG_ESSMHMM
	/*
	std::cout << "LogLikelihood = " << std::endl << LogLikelihood << std::endl;
	std::cout << "Postmix = " << std::endl << Postmix << std::endl;
	std::cout << "ExpNumTrans = " << std::endl << ExpNumTrans << std::endl;
	std::cout << "ExpNumVisits1 = " << std::endl << ExpNumVisits1 << std::endl;
	print4DMatrix(m);
	std::cout << "Ip = " << std::endl << Ip << std::endl;
	print4DMatrix(Op);
	std::cout << "Gamma = " << std::endl << Gamma << std::endl;
	print4DMatrix(Gamma2);
	*/
#endif


	for(size_t i = 0; i < Q; ++i) {
		for(size_t k = 0; k < M; ++k) {
			MatrixXf w = MatrixXf::Zero(1, T);
			for(size_t t = 0; t < T; ++t) {
				w(0, t) = Gamma2(t,0)(i,k);
			}// w = reshape(gamma2(i,k,:), [1 T]); % w(t) = w(i,k,t,l)
			MatrixXf wobs = obs.cwiseProduct(repmat<MatrixXf>(w, O, 1)); // wobs = obs .* repmat(w, [O 1]); % wobs(:,t) = w(t) * obs(:,t)
			m(k, 0).col(i) +=  wobs.rowwise().sum(); // m(:,i,k) = m(:,i,k) + sum(wobs, 2); % m(:) = sum_t w(t) obs(:,t)
			Op(i,k) += wobs * obs.transpose();//  op(:,:,i,k) = op(:,:,i,k) + wobs * obs'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
			MatrixXf t1 = ((wobs.cwiseProduct(obs)).rowwise().sum());
			MatrixXf t2 =  t1.colwise().sum();
			Ip(i,k) += t2(0,0);//ip(i,k) = ip(i,k) + sum(sum(wobs .* obs, 2)); % ip = sum_t w(t) * obs(:,t)' * obs(:,t)
		}
	}
/*
  for i=1:Q
    for k=1:M
      w = reshape(gamma2(i,k,:), [1 T]); % w(t) = w(i,k,t,l)
      wobs = obs .* repmat(w, [O 1]); % wobs(:,t) = w(t) * obs(:,t)
      m(:,i,k) = m(:,i,k) + sum(wobs, 2); % m(:) = sum_t w(t) obs(:,t)
      op(:,:,i,k) = op(:,:,i,k) + wobs * obs'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
      ip(i,k) = ip(i,k) + sum(sum(wobs .* obs, 2)); % ip = sum_t w(t) * obs(:,t)' * obs(:,t)
    end
  end
 */
#ifdef DEBUG_ESSMHMM
	std::cout << "LogLikelihood = " << std::endl << LogLikelihood << std::endl;
	std::cout << "Postmix = " << std::endl << Postmix << std::endl;
	std::cout << "ExpNumTrans = " << std::endl << ExpNumTrans << std::endl;
	std::cout << "ExpNumVisits1 = " << std::endl << ExpNumVisits1 << std::endl;
	print4DMatrix(m);
	std::cout << "Ip = " << std::endl << Ip << std::endl;
	print4DMatrix(Op);
	std::cout << "Gamma = " << std::endl << Gamma << std::endl;
	print4DMatrix(Gamma2);
#endif

}

double ESS_MHMM::getLogLikelihood() {return LogLikelihood;}
const MatrixXf& ESS_MHMM::getPostmix() const {return Postmix;}
const MatrixXf& ESS_MHMM::getExpNumTrans() const {return ExpNumTrans;}
const MatrixXf& ESS_MHMM::getExpNumVisits1() const{ return ExpNumVisits1;}
const MultiD& ESS_MHMM::getM() const {return m;}
const MatrixXf& ESS_MHMM::getIp() const {return Ip;}
const MultiD& ESS_MHMM::getOp() const {return Op;}

const MatrixXf& ESS_MHMM::getGamma() const {return fb.getGamma();}
const MultiD& ESS_MHMM::getGamma2() const {return fb.getGamma2();}

} /* namespace HMM */
