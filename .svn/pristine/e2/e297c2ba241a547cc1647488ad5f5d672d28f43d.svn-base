/*
 * MHMMEM.cpp
 *
 *  Created on: 21 апр. 2015 г.
 *      Author: kacher
 */

#include <iostream>
#include <limits>

#include "../../base/MHMMEM.h"
#include "../../base/MatlabUtils.h"
#include "../../base/MixgaussMstep.h"
#include "../../io/Util.h"

#include "../../base/Config.h"


namespace HMM {


MHMM_EM_Params::MHMM_EM_Params() : MaxIter(10), Threshold(1e-4), CovType("full"), AdjPrior(1),
		AdjTrans(1), AdjMix(1), AdjMu(1), AdjSigma(1) {}

void MHMM_EM_Params::setMaxIter(size_t max_iter) {MaxIter = max_iter;}
void MHMM_EM_Params::setThreshold(double threshold) {Threshold = threshold;}
void MHMM_EM_Params::setCovType(const std::string& cov_type) {CovType = cov_type;}

void MHMM_EM_Params::setAdjPrior(size_t adj_prior) {AdjPrior = adj_prior;}
void MHMM_EM_Params::setAdjTrans(size_t adj_trans) {AdjTrans = adj_trans;}
void MHMM_EM_Params::setAdjMix(size_t adj_mix) {AdjMix = adj_mix;}
void MHMM_EM_Params::setAdjMu(size_t adj_mu) {AdjMu = adj_mu;}
void MHMM_EM_Params::setAdjSigma(size_t adj_sigma) {AdjSigma = adj_sigma;}



MHMM_EM::MHMM_EM() {


}
MHMM_EM::MHMM_EM(
		const MatrixXf& data,
		const MatrixXf& prior,
		const MatrixXf& transmat,
		const MultiD& mu,
		const MultiD& sigma,
		const MatrixXf& mixmat,
		const MHMM_EM_Params& params) : Prior(prior), Transmat(transmat), Mu(mu), Sigma(sigma),
				Mixmat(mixmat), Params(params) {


	double previous_loglik = -std::numeric_limits<double>::infinity(); //previous_loglik = -inf;

	double loglik = 0.0; //loglik = 0;
	bool converged = false;  //converged = 0;
	size_t num_iter = 1; //num_iter = 1;

	size_t O = data.rows(); //O = size(data{1},1);
	size_t Q = Prior.size(); //Q = length(prior);

	if (isempty<MatrixXf>(Mixmat)) {
		Mixmat = MatrixXf::Ones(Q,1); // mixmat = ones(Q,1);
	}
	size_t M =  Mixmat.cols(); //M = size(mixmat,2);
	if (M == 1) {
		Params.AdjMix = 0;
	}

	CheckMatrixSize(data);


	while ((num_iter <= Params.MaxIter) && !converged) {
		//E step
#ifdef DEBUG_MHMM_EM
		/*
		std::cout << std::endl << "num_iter = " << num_iter << std::endl;
		std::cout << Prior << std::endl;
		std::cout << Transmat << std::endl;
		std::cout << Mixmat << std::endl;
		print4DMatrix(Mu);
		print4DMatrix(Sigma);
		std::cout << data << std::endl;*/
#endif
		ess_mhmm = ESS_MHMM(Prior, Transmat, Mixmat, Mu, Sigma, data);

		loglik = ess_mhmm.getLogLikelihood();

		if(Params.AdjPrior) {
			Prior = Normalize(ess_mhmm.getExpNumVisits1()).getNorm();
		}
		if(Params.AdjTrans) {
			Transmat = mkStochastic(ess_mhmm.getExpNumTrans());
		}
		if (Params.AdjMix) {
			Mixmat = mkStochastic(ess_mhmm.getPostmix());
		}


		if (Params.AdjMu || Params.AdjSigma) {
			MixgaussMstep mm(ess_mhmm.getPostmix(), ess_mhmm.getM(),ess_mhmm.getOp(), ess_mhmm.getIp(),  Params.CovType);
			// [mu2, Sigma2] = mixgauss_Mstep(postmix, m, op, ip, Paramscov_type);
			if(Params.AdjMu) {
				Mu = reshape2Dto4D(mm.getMu(), O, Q, M, 1);//mu = reshape(mu2, [O Q M]);
			}
			if (Params.AdjSigma) {
				Sigma = reshape4Dto4D(mm.getSigma(), O, O, Q, M );//Sigma = reshape(Sigma2, [O O Q M]);
			}
		}

		converged = EmConverged(loglik, previous_loglik);

		++num_iter;
		previous_loglik = loglik;
		LogLikelihood.push_back(loglik);

	}

#ifdef DEBUG_MHMM_EM
		std::cout << "Prior = " << std::endl << Prior  << std::endl;
		std::cout << "Transmat = " << std::endl << Transmat  << std::endl;
		std::cout << "Mu = " << std::endl;
		print4DMatrix(Mu);
		std::cout << "Sigma = " << std::endl;
		print4DMatrix(Sigma);
		std::cout << "Mixmat = " << std::endl << Mixmat  << std::endl;

		for(size_t i = 0; i < LogLikelihood.size(); ++i) {
			std::cout << "ll = " << LogLikelihood[i] << std::endl;
		}

		std::cout << "Gamma = " << std::endl << ess_mhmm.getGamma()  << std::endl;
		std::cout << "Gamma2 = " << std::endl;
		print4DMatrix(ess_mhmm.getGamma2());

#endif

}


void MHMM_EM::CheckMatrixSize(const MatrixXf& data) {
	size_t O = (size_t)data.rows();
	size_t Q = (size_t)Transmat.rows();
	size_t M = (size_t)Mixmat.cols();


	//check mu

	if (isempty<MultiD>(Mu))
		throw  std::runtime_error("Mu matrix must be not empty");

	if((size_t)Mu.rows() != M )
		throw  std::runtime_error("Wrong size Mu matrix");


	//check sigma
	if (isempty<MultiD>(Sigma))
		throw  std::runtime_error("Mu matrix must be not empty");

	if((size_t)Sigma.cols() != M || (size_t)Sigma.rows() != Q)
		throw  std::runtime_error("Wrong size Mu matrix");

	if((size_t)Sigma(0,0).cols() != O || (size_t)Sigma(0,0).rows() != O)
		throw  std::runtime_error("Wrong size Sigma matrix");

	//check prior
	if (isempty<MatrixXf>(Prior))
		throw  std::runtime_error("Prior matrix must be not empty");

	/*
	O - Dimension of the data point
	M - Number of mixture components used to model the data point distribution
	Q - Number of states in an HMM

	prior0 has size Q x 1
	transmat0 has size Q x Q
	mu0 has size O x Q x M
	Sigma0 has size O x O x Q x M
	mixmat0 has size Q x M; */

}

const std::vector<double>& MHMM_EM::getLogLikelihood() const {return LogLikelihood;}
const MatrixXf& MHMM_EM::getPrior() const {return Prior;}
const MatrixXf& MHMM_EM::getTransmat() const {return Transmat;}
const MultiD& MHMM_EM::getMu() const {return Mu;}
const MultiD& MHMM_EM::getSigma() const {return Sigma;}
const MatrixXf& MHMM_EM::getMixmat() const {return Mixmat;}

const MatrixXf& MHMM_EM::PostProbMC() const {return ess_mhmm.getGamma();}
const MultiD& MHMM_EM::PostProbMCandMOG() const {return ess_mhmm.getGamma2();}

bool MHMM_EM::EmConverged(double loglik, double previous_loglik) const {

	if (loglik - previous_loglik < -.001 ) // % allow for a little imprecision
		return 0;

	double delta_loglik = std::abs(loglik - previous_loglik);
	double avg_loglik = (fabs(loglik) + fabs(previous_loglik) + std::numeric_limits<double>::epsilon( ))/2.0;

	if ((delta_loglik / avg_loglik) < Params.Threshold)
		return 1;
	return 0;
}

} /* namespace HMM */
