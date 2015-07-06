/*
 * FwdBack.cpp
 *
 *  Created on: 22 апр. 2015 г.
 *      Author: kacher
 */

#include <iostream>

#include "../../base/FwdBack.h"
#include "../../base/MatlabUtils.h"
#include "../../io/Util.h"
#include "../../base/Config.h"

namespace HMM {

FwdBack::FwdBack() {
}

FwdBack::FwdBack(
		const MatrixXf& init_state_distrib,
		const MatrixXf& transmat,
		const MatrixXf& obslik,
		const MultiD& B2,
		const MatrixXf& _mixmat,
		size_t out_args
) : LogLikelihood(0.0) {
	size_t compute_xi = (out_args >= 5 ) ? 1 : 0;
	size_t compute_gamma2 = (out_args >= 6 ) ? 1 : 0;


	MultiD obslik2 = B2;
	MatrixXf mixmat = _mixmat;
	//size_t fwd_only = 0;
	//size_t scaled = 1;
	//size_t maximize = 0;
	size_t Q =  obslik.rows();
	size_t T = obslik.cols();
	compute_gamma2 = obslik2.size() ? 1 : 0;

	MatrixXf act = MatrixXf::Ones(1, T);
	/*if isempty(act)
 	 	 act = ones(1,T);
 	 	 transmat = { transmat } ;
	end*/

	MatrixXf scale = MatrixXf::Ones(1, T);
	/*
	 * % scale(t) = Pr(O(t) | O(1:t-1)) = 1/c(t) as defined by Rabiner (1989).
% Hence prod_t scale(t) = Pr(O(1)) Pr(O(2)|O(1)) Pr(O(3) | O(1:2)) ... = Pr(O(1), ... ,O(T))
% or log P = sum_t log scale(t).
% Rabiner suggests multiplying beta(t) by scale(t), but we can instead
% normalise beta(t) - the constants will cancel when we compute gamma.
	 *
	 */

	Alpha = MatrixXf::Zero(Q,T); //alpha = zeros(Q,T);
	Gamma = MatrixXf::Zero(Q,T); //gamma = zeros(Q,T);
	XiSummed = (compute_xi) ? MatrixXf::Zero(Q,Q) : MatrixXf();
	/*if compute_xi
	 xi_summed = zeros(Q,Q);
 else
	 xi_summed = [];
 end*/

	//%%%%%%%%% Forwards %%%%%%%%%%
#ifdef DEBUG_FWD_BACK
/*
	std::cout << "init_state_distrib = " << std::endl << init_state_distrib << std::endl;
	std::cout << "transmat = " << std::endl << transmat << std::endl;
	std::cout << "obslik = " << std::endl << obslik << std::endl;
	print4DMatrix(B2);
	std::cout << "_mixmat = " << std::endl << _mixmat << std::endl;*/
#endif



	size_t t = 0;
	Alpha.col(t) = init_state_distrib.col(0).cwiseProduct(obslik.col(t)); //alpha(:,1) = init_state_distrib(:) .* obslik(:,t);

	Normalize norm(Alpha.col(t)); //[alpha(:,t), scale(t)] = normalise(alpha(:,t));
	Alpha.col(t) = norm.getNorm();
	scale(0,t) =  norm.getNormalizingConst();


	for(size_t t = 1; t < T; ++t) {
		MatrixXf m;
		MatrixXf trans = transmat;

		m = trans.adjoint() * Alpha.col(t - 1); //m = trans' * alpha(:,t-1);
		Alpha.col(t) = m.col(0).cwiseProduct(obslik.col(t)); //alpha(:,t) = m(:) .* obslik(:,t);
		norm =  Normalize(Alpha.col(t));
		Alpha.col(t) = norm.getNorm();
		scale(0,t) =  norm.getNormalizingConst(); //[alpha(:,t), scale(t)] = normalise(alpha(:,t));
	}

	MatrixXi equal_zero = (scale.array() == MatrixXf::Zero(scale.rows(), scale.cols()).array()).cast<int>(); // scale==0

	if( equal_zero.sum() ) { //any(scale==0)
		LogLikelihood = -std::numeric_limits<double>::infinity();
	} else {
		LogLikelihood = scale.array().log().sum();//loglik = sum(log(scale));
	}

	//%%%%%%%%% Backwards %%%%%%%%%%

	Beta = MatrixXf::Zero(Q,T);
	size_t M = 0;
	if (compute_gamma2) {
		M = mixmat.cols();
		Gamma2 = MultiD(T, 1); //gamma2 = zeros(Q,M,T);
		for(size_t i = 0 ; i < T; ++i) {
			Gamma2(i,0) = MatrixXf::Zero(Q, M);
		}
	}
	Beta.col(T - 1) = MatrixXf::Ones(Q,1);//beta(:,T) = ones(Q,1);
	norm = Normalize(Alpha.col(T - 1).cwiseProduct(Beta.col(T - 1))); //normalise(alpha(:,T) .* beta(:,T));
	Gamma.col(T - 1) = norm.getNorm(); //gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));

	t = T - 1;


#ifdef DEBUG_FWD_BACK
/*
	std::cout << "Alpha = " << std::endl << Alpha << std::endl;
	std::cout << "Beta = " << std::endl << Beta << std::endl;
	std::cout << "Gamma = " << std::endl << Gamma << std::endl;*/
	//print4DMatrix(B2);
	//std::cout << "_mixmat = " << std::endl << _mixmat << std::endl;
#endif


	if (compute_gamma2) {
		MatrixXf tmp_matrix = obslik.col(t);
		MatrixXf eq_zero = (tmp_matrix.array() == MatrixXf::Zero(tmp_matrix.rows(), tmp_matrix.cols()).array()).cast<float>();
		MatrixXf denom = obslik.col(t) + eq_zero; //denom = obslik(:,t) + (obslik(:,t)==0); % replace 0s with 1s before dividing
#ifdef DEBUG_FWD_BACK
		/*std::cout << "tmp_matrix = " << std::endl << tmp_matrix << std::endl;
		std::cout << "eq_zero = " << std::endl << eq_zero << std::endl;
		std::cout << "denom = " << std::endl << denom << std::endl;*/
#endif
		Gamma2(t,0) = obslik2(t,0).cwiseProduct(mixmat).cwiseProduct(repmat<MatrixXf>(Gamma.col(t), 1, M)).cwiseQuotient(repmat<MatrixXf>(denom, 1, M));
		//gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom,  [1 M]);
	}


	for(int t = T - 2; t >= 0; --t) {
		MatrixXf b = Beta.col(t + 1).cwiseProduct(obslik.col(t+1));// b = beta(:,t+1) .* obslik(:,t+1);
		MatrixXf trans = transmat;

		Beta.col(t) = trans * b;


		norm = Normalize(Beta.col(t));
		Beta.col(t) = norm.getNorm(); //beta(:,t) = normalise(beta(:,t));

		norm = Normalize(Alpha.col(t).cwiseProduct(Beta.col(t)));
		Gamma.col(t) = norm.getNorm();//gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));

		norm = Normalize( trans.cwiseProduct(Alpha.col(t) * b.adjoint() ) );
		XiSummed += norm.getNorm();//xi_summed = xi_summed + normalise((trans .* (alpha(:,t) * b')));
		if (compute_gamma2) {
			MatrixXf tmp_matrix = obslik.col(t);
			MatrixXf eq_zero = (tmp_matrix.array() == MatrixXf::Zero(tmp_matrix.rows(), tmp_matrix.cols()).array()).cast<float>();
			MatrixXf denom = obslik.col(t) + eq_zero; //denom = obslik(:,t) + (obslik(:,t)==0); % replace 0s with 1s before dividing

			Gamma2(t,0) = obslik2(t,0).cwiseProduct(mixmat).cwiseProduct(repmat<MatrixXf>(Gamma.col(t), 1, M)).cwiseQuotient(repmat<MatrixXf>(denom, 1, M));
			//gamma2(:,:,t) = obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]) ./ repmat(denom,  [1 M]);
		}
	}
	/*
	 * % We now explain the equation for gamma2
% Let zt=y(1:t-1,t+1:T) be all observations except y(t)
% gamma2(Q,M,t) = P(Qt,Mt|yt,zt) = P(yt|Qt,Mt,zt) P(Qt,Mt|zt) / P(yt|zt)
%                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|zt) / P(yt|zt)
% Now gamma(Q,t) = P(Qt|yt,zt) = P(yt|Qt) P(Qt|zt) / P(yt|zt)
% hence
% P(Qt,Mt|yt,zt) = P(yt|Qt,Mt) P(Mt|Qt) [P(Qt|yt,zt) P(yt|zt) / P(yt|Qt)] / P(yt|zt)
%                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|yt,zt) / P(yt|Qt)
	 *
	 */

#ifdef DEBUG_FWD_BACK
	std::cout << "Alpha = " << std::endl << Alpha << std::endl;
	std::cout << "Beta = " << std::endl << Beta << std::endl;
	std::cout << "Gamma = " << std::endl << Gamma << std::endl;
	std::cout << "LogLikelihood = " << std::endl << LogLikelihood << std::endl;
	std::cout << "XiSummed = " << std::endl << XiSummed << std::endl;
	print4DMatrix(Gamma2);
#endif

}

const MatrixXf& FwdBack::getAlpha() const {return Alpha;}
const MatrixXf& FwdBack::getGamma() const {return Gamma;}
const MatrixXf& FwdBack::getBeta() const {return Beta;}
double FwdBack::getLogLik() const {return LogLikelihood;}
const MatrixXf& FwdBack::getXiSummed() const{return XiSummed;}
const MultiD& FwdBack::getGamma2() const {return Gamma2;}

} /* namespace HMM */
