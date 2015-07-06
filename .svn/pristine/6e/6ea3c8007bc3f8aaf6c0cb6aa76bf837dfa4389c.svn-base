/*
 * KMeansInit.h
 *
 *  Created on: 13 мая 2015 г.
 *      Author: kacher
 */

#ifndef KMEANSINIT_H_
#define KMEANSINIT_H_

#include "HMMTypes.h"
#include "Config.h"

namespace HMM {

class KMeansInit {
public:
	/*
	 * KMEANS	Trains a k means cluster model.
%
%	Description
%	 CENTRES = KMEANS(CENTRES, DATA, OPTIONS) uses the batch K-means
%	algorithm to set the centres of a cluster model. The matrix DATA
%	represents the data which is being clustered, with each row
%	corresponding to a vector. The sum of squares error function is used.
%	The point at which a local minimum is achieved is returned as
%	CENTRES.  The error value at that point is returned in OPTIONS(8).
%
%	[CENTRES, OPTIONS, POST, ERRLOG] = KMEANS(CENTRES, DATA, OPTIONS)
%	also returns the cluster number (in a one-of-N encoding) for each
%	data point in POST and a log of the error values after each cycle in
%	ERRLOG.    The optional parameters have the following
%	interpretations.
%
%	OPTIONS(1) is set to 1 to display error values; also logs error
%	values in the return argument ERRLOG. If OPTIONS(1) is set to 0, then
%	only warning messages are displayed.  If OPTIONS(1) is -1, then
%	nothing is displayed.
%
%	OPTIONS(2) is a measure of the absolute precision required for the
%	value of CENTRES at the solution.  If the absolute difference between
%	the values of CENTRES between two successive steps is less than
%	OPTIONS(2), then this condition is satisfied.
%
%	OPTIONS(3) is a measure of the precision required of the error
%	function at the solution.  If the absolute difference between the
%	error functions between two successive steps is less than OPTIONS(3),
%	then this condition is satisfied. Both this and the previous
%	condition must be satisfied for termination.
%
%	OPTIONS(14) is the maximum number of iterations; default 100.
	 *
	 */
#ifdef DEBUG_GMM
	KMeansInit(
			const MatrixXf& centres,
			const MatrixXf& data,
			const std::string& filename
		);
#else
	KMeansInit(
			const MatrixXf& centres,
			const MatrixXf& data
	);
#endif
	const MatrixXf& getCentres() const;
	const MatrixXf& getPost() const;
private:
	MatrixXf Centres;
	MatrixXf Post;
};

} /* namespace HMM */

#endif /* KMEANSINIT_H_ */
