/*
 * Utils.cpp
 *
 *  Created on: 25 мая 2015 г.
 *      Author: kacher
 */

#include <iostream>
#include <time.h>
#include "../../base/Util.h"
#include "../../io/Util.h"
#include "../../base/MatlabUtils.h"

namespace HMM {

MatrixXf InitPrior(size_t Q, size_t init_rnd, const std::string& filename) {
	srand (init_rnd);
	MatrixXf in_matrix = (filename == "") ? HMM::rand(Q, 1) : Read2DMatrix(filename);
	Normalize norm(in_matrix);
	return norm.getNorm(); //Prior = normalise(rand(Q,1));
}


MatrixXf InitTransmat(size_t Q, size_t init_rnd, const std::string& filename) {
	srand (init_rnd);
	MatrixXf in_matrix = (filename == "") ? HMM::rand(Q, Q) : Read2DMatrix(filename);
	return mkStochastic(in_matrix);
}

MatrixXf InitMixmat(size_t Q, size_t M, size_t init_rnd, const std::string& filename) {
	srand (init_rnd);
	MatrixXf in_matrix = (filename == "") ? HMM::rand(Q, M) : Read2DMatrix(filename);
	return mkStochastic(in_matrix);
}


} /* namespace HMM */
