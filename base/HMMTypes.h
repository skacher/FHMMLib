/*
 * HMMTypes.h
 *
 *  Created on: 20 апр. 2015 г.
 *      Author: kacher
 */

#ifndef HMMTYPES_H_
#define HMMTYPES_H_

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MatrixXi;

namespace HMM {
	typedef Eigen::Matrix< MatrixXf, Eigen::Dynamic, Eigen::Dynamic > MultiD; //4D
};



#endif /* HMMTYPES_H_ */
