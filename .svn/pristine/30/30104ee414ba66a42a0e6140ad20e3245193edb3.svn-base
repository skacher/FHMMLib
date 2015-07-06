/*
 * MatlabUtils.cpp
 *
 *  Created on: 21 àïð. 2015 ã.
 *      Author: kacher
 */
#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "../../base/MatlabUtils.h"
#include "../../base/Config.h"


namespace HMM {

double generateGaussianNoise(double mu, double sigma)
{
	const double epsilon = std::numeric_limits<double>::min();
	const double two_pi = 2.0*3.14159265358979323846;

	static double z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = ::rand() * (1.0 / RAND_MAX);
	   u2 = ::rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}


Normalize::Normalize(const MatrixXf& in) : Norm(in) {
	NormalizingConst = Norm.sum();
	double s = NormalizingConst + double(NormalizingConst == 0);
	Norm /= s;
}
double Normalize::getNormalizingConst() const {return NormalizingConst;}
const MatrixXf& Normalize::getNorm() const {return Norm;}


MatrixXf mkStochastic(const MatrixXf& t) {
	if(t.cols() == 1 || t.rows() == 1) { //% isvector
		return Normalize(t).getNorm();
	}
	else { //case matrix
		MatrixXf Z = t.rowwise().sum(); // Z = sum(T,2);
		MatrixXf eq_zero = (Z.array() == MatrixXf::Zero(Z.rows(), Z.cols()).array()).cast<float>();
		MatrixXf S = Z + eq_zero; // S = Z + (Z==0);
		MatrixXf norm = repmat<MatrixXf>(S, 1, t.cols()); //repmat(S, 1, size(T,2));
		MatrixXf result = t.cwiseQuotient(norm); // T ./ norm;
		return result;
	}
}



MultiD reshape2Dto4D(const MatrixXf& in, size_t d1, size_t d2, size_t d3, size_t d4) {
	if((size_t)in.size() != d1 * d2 * d3 * d4)
		 throw  std::runtime_error("reshape2Dto4D: matrix dimensions must agree "); ;//êèíóòü ýêñåïøåí

	MultiD result(d3, d4);
	for(size_t r = 0; r < d3; ++r) {
		for(size_t c = 0; c < d4; ++c) {
			result(r, c) = MatrixXf::Zero(d1,d2);
		}
	}

	size_t i = 0;
	size_t j = 0;
	size_t r = 0;
	size_t c = 0;
	for(size_t col = 0; col < (size_t)in.cols(); ++col) {
		for(size_t row = 0; row < (size_t)in.rows(); ++row) {
			result(i,j)(r,c) = in(row, col);
			++r;
			if(r == d1) {
				r = 0;
				++c;
				if (c == d2) {
					c = 0;
					++i;
					if(i == d3) {
						i = 0;
						++j;
					}
				}
			}
		}
	}
	return result;
}

MultiD reshape4Dto4D(const MultiD& in, size_t d1, size_t d2, size_t d3, size_t d4) {
	if((size_t)in.size() * (size_t)in(0,0).size() != d1 * d2 * d3 * d4) {
		throw  std::runtime_error("reshape4Dto4D: count of matrix's elements must be the same");
	}

	MultiD result(d3, d4);
	for(size_t r = 0; r < d3; ++r) {
		for(size_t c = 0; c < d4; ++c) {
			result(r, c) = MatrixXf::Zero(d1,d2);
		}
	}

	size_t dst_i = 0;
	size_t dst_j = 0;
	size_t dst_r = 0;
	size_t dst_c = 0;

	for(size_t src_i = 0; src_i < (size_t)in.cols(); ++src_i) {
		for(size_t src_j = 0; src_j < (size_t)in.rows(); ++src_j) {
			MatrixXf t = in(src_j, src_i);
			for(size_t src_c = 0; src_c < (size_t)t.cols(); ++src_c) {
				for(size_t src_r = 0; src_r < (size_t)t.rows(); ++src_r) {
					result(dst_i, dst_j)(dst_r, dst_c) = t(src_r, src_c);
					++dst_r;
					if(dst_r == d1) {
						dst_r = 0;
						++dst_c;
						if(dst_c == d2) {
							dst_c = 0;
							++dst_i;
							if(dst_i == d3) {
								dst_i = 0;
								++dst_j;
							}
						}
					}
				}
			}
		}
	}
	return result;
}


MatrixXf gaussianProb(const MatrixXf& x, const MatrixXf& m, const MatrixXf& C) {
	/*if length(m)==1 % scalar
	  x = x(:)';
	end*/
	//IS IT POSSIBLE ????

	size_t d = x.rows(); //[d N] = size(x);
	size_t N = x.cols();

	MatrixXf M = m * MatrixXf::Ones(1,N);

#ifdef DEBUG_GAUSSIAN_PROB
	//std::cout << "d = " << d <<  " " << "N = " << N << std::endl;
	//std::cout << "m = " << m << std::endl;
	//std::cout << "M = " << std::endl;
	//std::cout << M << std::endl;
#endif

	double denom = std::pow(2 * M_PI, double(d) / 2.0) * std::sqrt(std::abs(C.determinant()));

	MatrixXf t = (x - M);
	MatrixXf tmp = (t.transpose() * C.inverse()).cwiseProduct(t.transpose()); // ÏÐÀÂÈËÜÍÛÉ ËÈ ÏÐÈÎÐÈÒÅÒ ÎÏÅÐÀÖÈÉ!?
	MatrixXf mahal = -0.5 * tmp.rowwise().sum();
	return mahal.array().exp() / (denom + std::numeric_limits<double>::epsilon( ));
}

MatrixXf rand(size_t rows, size_t cols) {
	return (MatrixXf::Random(rows, cols).array() + 1.0) / 2.0;
}

MatrixXf randn(size_t rows, size_t cols) {
	MatrixXf result = MatrixXf::Zero(rows, cols);
	for(size_t row  = 0; row <rows; ++row) {
		for(size_t col  = 0; col <cols; ++col) {
			result(row, col) = generateGaussianNoise(0.0, 1.0);
		}
	}
	return result;
}


std::vector<size_t> randperm(size_t n) {
	std::vector<size_t> result(n, 0);
	for(size_t i = 0; i < n; ++i) {
		result[i] = i;
	}
	std::random_shuffle ( result.begin(), result.end() );
	return result;
}

MatrixXf dist2(const MatrixXf& x, const MatrixXf& c) {

	size_t ndata = x.rows();
	size_t dimx = x.cols();//[ndata, dimx] = size(x);

	size_t ncentres = c.rows();
	size_t dimc = c.cols();  //[ncentres, dimc] = size(c);

	if(dimx != dimc)
		throw  std::runtime_error("dist2: Data dimension does not match dimension of centres");

	MatrixXf sq_x = x.array().square();
	MatrixXf sq_c = c.array().square();
	MatrixXf sq_x_transp = sq_x.adjoint();
	MatrixXf sq_c_transp = sq_c.adjoint();



	MatrixXf a = (MatrixXf::Ones(ncentres, 1) * sq_x_transp.colwise().sum()).adjoint();
	MatrixXf b = (MatrixXf::Ones(ndata, 1) * sq_c_transp.colwise().sum());
	MatrixXf f = 2 * (x * (c.adjoint()));

	MatrixXf n2 = a + b - f;

	//% Rounding errors occasionally cause negative entries in n2
	//if any(any(n2<0))  n2(n2<0) = 0;	end
	for(size_t row = 0; row < (size_t)n2.rows(); ++row) {
		for(size_t col = 0; col < (size_t)n2.cols(); ++col) {
			if(n2(row, col) < 0)
				n2(row, col) = 0;
		}
	}

	return n2;
}

void min_with_index(const MatrixXf& x, int dim, MatrixXf& values, MatrixXf& index) {
	assert(dim==1||dim==2);
	// output size

	size_t n = (size_t)(dim == 1 ? x.cols() : x.rows());

	// resize output
	values = MatrixXf::Zero(n,1);
	index = MatrixXf::Zero(n,1);

	// loop over dimension opposite of dim
	for(size_t j = 0; j < n; j++) {
		MatrixXf::Index fake;
		MatrixXf::Index i;
		float m;

		if(dim == 1) {
			m = x.col(j).minCoeff(&i, &fake);
		} else {
			m = x.row(j).minCoeff(&fake, &i);
		}
		values(j, 0) = m;
		index(j, 0) = i;
	}

}

void max_with_index(const MatrixXf& x, int dim, MatrixXf& values, MatrixXf& index) {
	assert(dim==1||dim==2);
	// output size

	size_t n = (size_t)(dim == 1 ? x.cols() : x.rows());

	// resize output
	values = MatrixXf::Zero(n,1);
	index = MatrixXf::Zero(n,1);

	// loop over dimension opposite of dim
	for(size_t j = 0; j < n; j++) {
		MatrixXf::Index fake;
		MatrixXf::Index i;
		float m;

		if(dim == 1) {
			m = x.col(j).maxCoeff(&i, &fake);
		} else {
			m = x.row(j).maxCoeff(&fake, &i);
		}
		values(j, 0) = m;
		index(j, 0) = i;
	}

}

double eps(double in) {
#ifdef _WIN32
	return double(in) - _nextafter(double(in),  std::numeric_limits<double>::epsilon());
#elif _WIN64
	return double(in) - _nextafter(double(in),  std::numeric_limits<double>::epsilon());
#else 
	return double(in) - ::nextafter(double(in),  std::numeric_limits<double>::epsilon());
#endif
}

size_t rank(const MatrixXf& in) {
	Eigen::JacobiSVD<MatrixXf> svd(in);
	MatrixXf s = svd.singularValues();

	double tol = (in.cols() > in.rows() ? in.cols() : in.rows()) * eps(s.maxCoeff());

	MatrixXf tmp = (s.array() > tol).cast<float>();

	return tmp.sum();
}


} // //end HMM namesapace
