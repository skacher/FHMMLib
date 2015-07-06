/*
 * MatlabUtils.h
 *
 *  Created on: 21 апр. 2015 г.
 *      Author: kacher
 */

#ifndef MATLABUTILS_H_
#define MATLABUTILS_H_

#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

#include "HMMTypes.h"

using Eigen::MatrixXf;

namespace HMM {

double generateGaussianNoise(double mu, double sigma);


//check if matrix is empty
template <class T>
bool isempty(const T& t) {
	return t.cols() == 0 && t.rows() == 0;
}

//simple matlab repmat
template <class T>
T repmat(const T& t, size_t d1, size_t d2) {
	return t.replicate(d1,d2);
}

template <class T, class M>
M repmat(const T& t, size_t d1, size_t d2, size_t d3, size_t d4) {
	T tmp = t.replicate(d1,d2);
	M result = M(d3, d4);
	for(size_t row = 0; row < d3; ++row) {
		for(size_t col = 0; col < d4; ++col) {
			result(row, col) = tmp;
		}
	}
	return result;
}


class Normalize
{
public:
	Normalize(const MatrixXf& in);
	double getNormalizingConst() const;
	const MatrixXf& getNorm() const;
public:
	double NormalizingConst;
	MatrixXf Norm;
};

MatrixXf mkStochastic(const MatrixXf& t);

//convert 2D matrix to 4D
MultiD reshape2Dto4D(const MatrixXf& in, size_t d1, size_t d2, size_t d3, size_t d4);
//convert 4D matrix to 4D
MultiD reshape4Dto4D(const MultiD& in, size_t d1, size_t d2, size_t d3, size_t d4);

/*% GAUSSIAN_PROB Evaluate a multivariate Gaussian density.
% p = gaussian_prob(X, m, C)
% p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector

% p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevents underflow).
%
% If X has size dxN, then p has size Nx1, where N = number of examples
 */
MatrixXf gaussianProb(const MatrixXf& x, const MatrixXf& m, const MatrixXf& C);

MatrixXf rand(size_t rows, size_t cols);

MatrixXf randn(size_t rows, size_t cols);

std::vector<size_t> randperm(size_t n);

/*
 * %   DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
 */

MatrixXf dist2(const MatrixXf& x, const MatrixXf& c);

void min_with_index(const MatrixXf& x, int dim, MatrixXf& values, MatrixXf& index);
void max_with_index(const MatrixXf& x, int dim, MatrixXf& values, MatrixXf& index);

template <class T>
T varaince(const T& in) {
	T mean     = in.colwise().mean();
	T sq = in.array().pow(2.0);
	T variance = (sq.colwise().mean()).array() - mean.array()*mean.array();
	return variance;
}

template <class T>
T cov(const T& in) {
	MatrixXf mean  = in.colwise().mean();

	MatrixXf xc = in;
	for(size_t i = 0; i < (size_t)in.rows(); ++i)
		xc.row(i) = xc.row(i) - mean.row(0);

	T result = ( xc.adjoint() * xc ) / (in.rows() - 1); // here might be a problem!
	return result;
}



double eps(double in);


size_t rank(const MatrixXf& in);

} //end HMM namesapace

#endif /* MATLABUTILS_H_ */
