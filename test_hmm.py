from pylab import *
from hmm import HMM, init_prior, init_transmat, init_mixmat, MxGaussInit
import random

random.seed()

dimensions_count = 2  # O
mixture_components_count = 1 # M
steps_count = 100
states_count = 4 #Q

prior = init_prior(states_count, random.randint(0, 10**6) );
transmat = init_transmat(states_count, random.randint(0, 10**6) );
mixmat = init_mixmat(states_count,mixture_components_count, random.randint(0, 10**6) );
#data = randn(steps_count, dimensions_count)
data = randn(dimensions_count,steps_count)

mgi = MxGaussInit(states_count, mixture_components_count, dimensions_count, data, 'full', 'rnd')

print "Sigma"
sigma = mgi.sigma
print sigma


print "mu"
mu = mgi.mu
print mu



#prior = ones((1, states_count)) * 0.5
#data = randn(dimensions_count, steps_count)
#transmat = np.ones((states_count, states_count)) / states_count
#mixmat = ones((1, mixture_components_count)) / mixture_components_count
#mu = np.ones((mixture_components_count, 1, dimensions_count, states_count))
#sigma = np.ones((states_count, mixture_components_count,  dimensions_count, dimensions_count))
model = HMM(threshold=1e-4, adjustSigma=1, adjustMu=1, 
	        adjustPrior=1, adjustTransmition=1, 
	        adjustMix=4, covarianceType='full', maxIter=10)
model.fit(data, prior, transmat, mixmat, mu, sigma)
print model.sigma
print model.mu
print model.transmissions
print model.mix
print model.prior

print model.post_prob_mc
print model.post_prob_mcand_mog
