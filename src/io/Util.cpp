/*
 * util.cpp
 *
 *  Created on: 20 апр. 2015 г.
 *      Author: kacher
 */

#include <iostream>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <Eigen/Dense>
using Eigen::MatrixXf;

//#include "../../io/util.h"
#include "../../base/MatlabUtils.h"


namespace HMM {

/*in - file name with matrix description
 * d1 - dimension 1
 * d2 - dimension 2
 * d3 - dimension 3
 * d4 - dimension 4
 * v1\nv2\n...vn
*/
MultiD Read4DMatrix(const std::string& filename) {

	std::ifstream fin(filename.c_str(), std::ios::in);
	if(!fin) { throw  std::runtime_error("Cannot open this file: " + filename); }

	std::string line;

	size_t line_cnt = 0;
	size_t d1 = 0;
	size_t d2 = 0;
	size_t d3 = 0;
	size_t d4 = 0;
	size_t row = 0;

	MatrixXf D2; //2d matrix
	while(! fin.eof()) {
		std::getline(fin, line);
		if (line == "") // matlab writes empty string at the end of file
			continue;
		if (line_cnt > 3) {
			D2(row++, 0) = std::atof(line.c_str());
		}
		else if(line_cnt == 0)
			d1 = std::atoi(line.c_str());
		else if(line_cnt == 1)
			d2 = std::atoi(line.c_str());
		else if(line_cnt == 2)
			d3 = std::atoi(line.c_str());
		else if(line_cnt == 3) { //now we read all dimensions
			d4 = std::atoi(line.c_str());
			D2 = MatrixXf::Zero(d1 * d2 * d3 * d4, 1);
		}
		++line_cnt;
	}

#ifdef DEBUG
	std::cout << D2 << std::endl;
#endif

	return reshape2Dto4D(D2, d1, d2, d3, d4);

}

std::vector<double> ParseRow(const std::string& in) {
	std::vector<double> result;

	std::istringstream sin(in);
	std::string row;
	while(getline(sin, row, '\t')) {
		result.push_back(std::atof(row.c_str()));
	}
	return result;

}

//reads tab-separated file
MatrixXf Read2DMatrix(const std::string& filename) {
	std::ifstream fin(filename.c_str(), std::ios::in);
	if(!fin) { throw  std::runtime_error("Cannot open this file: " + filename); }

	std::vector< std::vector<double> > rows;
	std::string line;

	while(! fin.eof()) {
		std::getline(fin, line);
		if (line == "") // matlab writes empty string at the end of file
			continue;

		rows.push_back(ParseRow(line));
	}
	//make check if all vectors length are the same
	if (rows.empty())
		return MatrixXf();

	MatrixXf D2 = MatrixXf(rows.size(), rows[0].size());
	for (size_t r = 0; r < rows.size(); ++r) {
		for(size_t c = 0; c < rows[0].size(); ++c) {
			D2(r,c) = rows[r][c];
		}
	}
	return D2;
}

void Save4DMatrix(const MultiD& in, const std::string& filename) {
	std::ofstream fout(filename.c_str());
	fout.setf(std::ios::fixed, std:: ios::floatfield);
	fout.precision(15);

	size_t outer_rows = (size_t)in.rows();
	size_t outer_cols = (size_t)in.cols();

	if(outer_rows == 0 || outer_cols == 0)  {
		fout.close();
		return;
	}
	size_t inner_rows = (size_t)in(0,0).rows();
	size_t inner_cols = (size_t)in(0,0).cols();

	fout << inner_rows << "\n" << inner_cols << "\n" << outer_rows << "\n" << outer_cols << std::endl;


	for(size_t ocol = 0; ocol < outer_cols; ++ocol) {
		for(size_t orow = 0; orow < outer_rows; ++orow) {
			const MatrixXf& t = in(orow,ocol);

			for(size_t row = 0; row < inner_rows; ++row) {
				for(size_t col = 0; col < inner_cols; ++col) {
					fout << t(row,col) << std::endl;
				}
			}
		}
	}

	fout.close();
}
void Save2DMatrix(const MatrixXf& in, const std::string& filename) {
	std::ofstream fout(filename.c_str());
	fout.setf(std::ios::fixed, std:: ios::floatfield);
	fout.precision(15);

	size_t rows = (size_t)in.rows();
	size_t cols = (size_t)in.cols();


	for(size_t row = 0; row < rows; ++row) {
		for(size_t col = 0; col < cols; ++col) {
			if(col)  {
				fout << "\t" << in(row,col);
			} else {fout << in(row,col);}
		}
		fout << std::endl;
	}

	fout.close();
}

void print4DMatrix(const MultiD& in) {
	size_t rows = (size_t)in.rows();
	size_t cols = (size_t)in.cols();
	std::cout << std::endl;
	for(size_t r = 0; r < rows; ++r) {
		for(size_t c = 0; c < cols; ++c) {
			std::cout << "(" << r << "," << c << ")" << std::endl;
			std::cout << in(r,c) << std::endl;
		}
	}
	std::cout << std::endl;
}

} //end HMM namesapace


