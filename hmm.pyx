#clib m
import sys
from libc.stdlib cimport malloc, free
from collections import namedtuple
from cpython.string cimport PyString_AsString
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
from cpython cimport bool
cimport numpy as np

cdef extern from "eigen3/Eigen/Dense" namespace "Eigen":
    cdef cppclass MatrixXf:
        MatrixXf() except +
        float& element "operator()"(int row,int col)
        void resize(int row, int col)
        int cols()
        int rows()

cdef extern from "./base/HMMTypes.h" namespace "HMM":
    cdef cppclass MultiD:
        MultiD() except +
        MultiD(int row, int col) except +
        MatrixXf& element "operator()"(int row,int col)
        void resize(int row, int col)
        int cols()
        int rows()

cdef extern from "./base/MHMMEM.h" namespace "HMM":
    cdef cppclass MHMM_EM_Params:
        MHMM_EM_Params() except +
        void setMaxIter(unsigned int max_iter)
        void setThreshold(double max_iter)
        void setCovType(const string& cov_type)

        void setAdjPrior(unsigned int adj_prior)
        void setAdjTrans(unsigned int adj_trans)
        void setAdjMix(unsigned int adj_mix)
        void setAdjMu(unsigned int adj_mu)
        void setAdjSigma(unsigned int adj_sigma)
    
    cdef cppclass MHMM_EM:
        MHMM_EM() except +
        MHMM_EM(
            const MatrixXf& data,
            const MatrixXf& prior,
            const MatrixXf& transmat,
            const MultiD& mu,
            const MultiD& Sigma,
            const MatrixXf& mixmat,
            const MHMM_EM_Params& params) except +
        const vector[double]& getLogLikelihood() const
        const MatrixXf& getPrior() const
        const MatrixXf& getTransmat() const
        const MultiD& getMu() const
        const MultiD& getSigma() const
        const MatrixXf& getMixmat() const
        const MultiD& PostProbMCandMOG() const
        const MatrixXf& PostProbMC() const

cdef extern from "./base/MixGaussInit.h" namespace "HMM":
        cdef cppclass MixGaussInit:
                MixGaussInit()
                MixGaussInit(int q, int m, int o, const MatrixXf& obs, string& cov_type, string& mode) except +
                const MultiD& getSigma() const
                const MultiD& getMu() const

class MxGaussInit:
  def __init__(self, q, m, o, obs, cov_type, mode):
    cdef MatrixXf ob = numpy_to_eigen(obs)
    cdef MixGaussInit mgi = MixGaussInit(q, m, o, ob, cov_type, mode)

    cdef MultiD sigma = mgi.getSigma()
    cdef MultiD mu = mgi.getMu()

    self.sigma = multi_to_numpy(sigma)
    self.mu = multi_to_numpy(mu)

cdef extern from "./base/Util.h" namespace "HMM":
        cdef MatrixXf InitPrior(int Q)
        cdef MatrixXf InitTransmat(int Q)
        cdef MatrixXf InitMixmat(int Q, int M)

def init_prior(Q):
        cdef MatrixXf m = InitPrior(Q)
        cdef np.ndarray newPrior = eigen_to_numpy(m)
        return newPrior

def init_transmat(Q):
        cdef MatrixXf m = InitTransmat(Q)
        return eigen_to_numpy(m)


def init_mixmat(Q,M):
        cdef MatrixXf m = InitMixmat(Q,M)
        return eigen_to_numpy(m)

cdef void set_matrix_element(MatrixXf& matrix, int row_index,
	                         int col_index, float element):
    cdef float* elem_position = &(matrix.element(row_index, col_index))
    elem_position[0] = element



cdef void set_matrix_element_multi(MultiD& matrix, int row_index,
	                    int col_index, MatrixXf element):
    cdef MatrixXf* elem_position = &(matrix.element(row_index, col_index))
    elem_position[0] = element

cdef MatrixXf numpy_to_eigen(array):
    cdef MatrixXf matrix = MatrixXf()
    matrix.resize(array.shape[0], array.shape[1])
    cdef int row_index, col_index
    for row_index in range(array.shape[0]):
        for col_index in range(array.shape[1]):
            set_matrix_element(matrix, row_index,
            	col_index, array[row_index, col_index])
    return matrix

cdef eigen_to_numpy(MatrixXf matrix):
    cdef int row_count, col_count
    row_count = matrix.rows()
    col_count = matrix.cols()
    array = np.zeros((row_count, col_count))
    for row_index in range(row_count):
        for col_index in range(col_count):
            array[row_index, col_index] = matrix.element(row_index, col_index)
    return array

cdef MultiD numpy_to_eigen_multy(array):
    cdef MultiD matrix = MultiD()
    cdef MatrixXf matrix_elem
    matrix.resize(array.shape[0], array.shape[1])
    cdef int row_index, col_index
    for row_index in range(array.shape[0]):
        for col_index in range(array.shape[1]):
            matrix_elem = numpy_to_eigen(array[row_index, col_index])
            set_matrix_element_multi(matrix, row_index, col_index, matrix_elem)
    return matrix


cdef multi_to_numpy(MultiD matrix):

    cdef int row_count, col_count
    cdef MatrixXf inner_matrix = matrix.element(0, 0)
    cdef int inner_row_count, inner_col_count
    inner_row_count = inner_matrix.rows()
    inner_col_count = inner_matrix.cols()
    row_count = matrix.rows()
    col_count = matrix.cols()
    array = np.zeros((row_count, col_count, inner_row_count, inner_col_count))
    for row_index in range(row_count):
        for col_index in range(col_count):
            array[row_index, col_index, :, :] = eigen_to_numpy(matrix.element(row_index, col_index))
    return array


def test():
    cdef MHMM_EM_Params params = MHMM_EM_Params()
    params.setThreshold(1)
    params.setAdjSigma(1)
    params.setAdjMu(1)
    params.setCovType('full')
    params.setMaxIter(1)
    params.setAdjPrior(1)
    params.setAdjTrans(1)
    params.setAdjMix(1)
    cdef MatrixXf prior = numpy_to_eigen(np.array([[0.5, 0.5]]))
    cdef MatrixXf data = numpy_to_eigen(np.array([[0, 0, 0, 0, 0, 0,0]]))
    cdef MatrixXf transmat = numpy_to_eigen(np.array([[0.5, 0.5], [0.5, 0.5]]))
    cdef MatrixXf mixmat = numpy_to_eigen(np.array([[0.5, 0.5]]))
    cdef MultiD mu = numpy_to_eigen_multy(np.ones((2, 1, 1, 1)))
    cdef MultiD sigma = numpy_to_eigen_multy(np.ones((2, 2, 1, 1)))
    cdef MHMM_EM optimizer = MHMM_EM(data, prior, transmat, mu, sigma, mixmat, params)
    cdef MultiD new_mu =  optimizer.getMu()
    print multi_to_numpy(new_mu)

def low_level_fit(unsigned int threshold, unsigned int adjustSigma,
                  unsigned int adjustMu, unsigned int adjustPrior,
                  unsigned int adjustTransmition, unsigned int adjustMix,
                  string covarianceType, unsigned int maxIter,
                  np.ndarray initMix,
                  np.ndarray trainData,
                  np.ndarray initPrior,
                  np.ndarray initTrans,
                  np.ndarray initMu,
                  np.ndarray initSigma):
    """
    Transforms numpy array to eigen format
    train model and return results as namedtuple
    """
    cdef MHMM_EM_Params params = MHMM_EM_Params()
    params.setThreshold(threshold)
    params.setAdjSigma(adjustSigma)
    params.setAdjMu(adjustMu)
    params.setCovType(covarianceType)
    params.setMaxIter(maxIter)
    params.setAdjPrior(adjustPrior)
    params.setAdjTrans(adjustTransmition)
    params.setAdjMix(adjustMix)
    cdef MatrixXf trainDataEigen = numpy_to_eigen(trainData)
    cdef MatrixXf initPriorEigen = numpy_to_eigen(initPrior)
    cdef MatrixXf initTransEigen = numpy_to_eigen(initTrans)
    cdef MatrixXf initMixEigen = numpy_to_eigen(initMix)
    cdef MultiD initMuEigen = numpy_to_eigen_multy(initMu)
    cdef MultiD initSigmaEigen = numpy_to_eigen_multy(initSigma)
    cdef MHMM_EM optimizer = MHMM_EM(trainDataEigen, initPriorEigen, 
                                    initTransEigen, initMuEigen, 
                                    initSigmaEigen, initMixEigen, params)

    cdef MatrixXf newPriorEigen = optimizer.getPrior()
    cdef MatrixXf newTransEigen = optimizer.getTransmat()
    cdef MatrixXf newMixEigen = optimizer.getMixmat()
    cdef MultiD newMuEigen = optimizer.getMu()
    cdef MultiD newSigmaEigen = optimizer.getSigma()
    cdef vector[double]& liklihood = optimizer.getLogLikelihood()


    cdef MultiD postProbMCandMOGEigen = optimizer.PostProbMCandMOG() if initMixEigen.cols() > 1 else MultiD(0,0)  
    cdef MatrixXf postProbMCEigen = optimizer.PostProbMC()
    cdef np.ndarray newPrior = eigen_to_numpy(newPriorEigen)
    cdef np.ndarray newTrans = eigen_to_numpy(newTransEigen)
    cdef np.ndarray newMix = eigen_to_numpy(newMixEigen)
    cdef np.ndarray newMu = multi_to_numpy(newMuEigen)
    cdef np.ndarray newSigma = multi_to_numpy(newSigmaEigen)

    cdef np.ndarray postProbMCandMOG = multi_to_numpy(postProbMCandMOGEigen) if initMixEigen.cols() > 1 else np.zeros(0)
    cdef np.ndarray postProbMC = eigen_to_numpy(postProbMCEigen)

    model_parameters = namedtuple('model_parameters', ['prior', 'transmission', 
                                                            'mix', 'mu', 'sigma', 'post_prob_mc', 'post_prob_mcand_mog'])
    return model_parameters(newPrior, newTrans, newMix, newMu, newSigma, postProbMC, postProbMCandMOG)


class HMM(object):
    """
    This front end class provides  
    interface for training HMM model and 
    getting acces to its parameters
    """

    def __init__(self, threshold, adjustSigma, adjustMu, 
                 adjustPrior, adjustTransmition, adjustMix,
                 covarianceType, maxIter):
        self.threshold = threshold
        self.adjustSigma = adjustSigma
        self.adjustMu = adjustMu
        self.adjustPrior = adjustPrior
        self.adjustTransmition = adjustTransmition
        self.adjustMix = adjustMix
        self.covarianceType = covarianceType
        self.maxIter = maxIter
    
    def fit(self, train_data, initPrior, initTrans, initMix, initMu, initSigma):
        new_parameters = low_level_fit(self.threshold, self.adjustSigma,
                                       self.adjustMu, self.adjustPrior,
                                       self.adjustTransmition, self.adjustMix,
                                       self.covarianceType, self.maxIter,
                                       initMix, train_data, initPrior, initTrans,
                                       initMu, initSigma)
        self.mix = new_parameters.mix
        self.prior = new_parameters.prior
        self.transmissions = new_parameters.transmission
        self.mu = new_parameters.mu
        self.sigma = new_parameters.sigma
        self.post_prob_mc = new_parameters.post_prob_mc
        self.post_prob_mcand_mog = new_parameters.post_prob_mcand_mog
