#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
%% Bayesian parameter estimation for rupture occurrence model: (Ceferino et al. BSSA (2018))
% Developed by: Luis Ceferino
% Date: 02/21/2017
---------------------------------------------
% Script to run Bayesian inference using MCMC
---------------------------------------------
"""
# Required imports
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.stats import mvn
from scipy.stats import norm
import pickle

###################### DATA #############################
np.random.seed(1)
# Load variables
filename = 'Rupture_data_historic'
f = open(filename, 'rb')
N = pickle.load(f) # Number of area units
n_years = pickle.load(f) # Number of years of rupture data
Rupture_history = pickle.load(f) # Rows: number of years, N: number of units (from south to north)
Years_no_EQ = pickle.load(f) # Rows: number of years, N: number of units
loc_rup_units = pickle.load(f) # Rupture units coordinates
f.close()

#Useful expressions
sqrt_Years_no_EQ = np.sqrt(Years_no_EQ)
inv_sqrt_Years_no_EQ = 1/np.sqrt(Years_no_EQ)
sqrt_Years_no_EQ_plus_1 = np.sqrt(Years_no_EQ+1)
# Distance matrix
x_temp, y_temp = np.meshgrid(loc_rup_units, loc_rup_units)
sq_dist_matrix = np.square(x_temp-y_temp)
ind_l = np.tril_indices(N,k=-1) ## select lower triangular of distance matrix (Input for correlation later)
sq_dist_low_sq_m = sq_dist_matrix[ind_l]


####################### Posterior MCMC Input #####################
M = 100 # Number of samples
B = 20 # Burning period
T_p = 2*N+1 # Number of parameters
parameter_samples = np.zeros((M,T_p)) # Column: N aplha parameters, N mu parameters, and gamma
    									  # Row: M samples
parameter_likelihood = np.zeros(M) # Column: N aplha parameters, N mu parameters, and gamma
    									  # Row: M samples
parameter_samples[0,:] = np.array([1]*N + [150]*N + [300]) #Initial value

######################## Define random walk in MCMC ###################
sigma_walk = np.array([.04]*N + [4]*N + [7])*5 # Defines the standard deviation of the random walk
######################## Prior ##################################
#median_parameter_prior = np.array([1]*N + [50]*N + [200])
median_parameter_prior = np.array([.92, 1.22, 0.68, .73, .99, 1.08, 0.75, .96,\
                                    320, 172, 194, 97, 114, 110, 144, 146, 290]) # From the previous MLE analysis (ICOSAAr)

s = 0.3 # Same log stand dev for all
#log_std_parameter_prior = np.array([1]*N + [1]*N + [1])

################# Compute MCMC ##########################
prev_post_log_lik = -float('inf') # Always accept initial sample
A_rate = 0
for i in range(1,M):
    print "Sample: ", i
    sample_prop = parameter_samples[i-1,:] + np.random.normal(0, 1, T_p)*sigma_walk
    #print sample_prop
    ####### Prior ######
    #print np.sum((np.square(np.log(sample_prop/median_parameter_prior))))/(2*s*s)
    if (sample_prop >= 0).all():
        log_prior = np.sum((-np.log(sample_prop)-np.log(s)) - \
        					np.square(np.log(sample_prop/median_parameter_prior))/(2*s*s))
        ####### Likelihood of data ######
         # Find marginal
        sqrt_u = np.sqrt(sample_prop[N:2*N])
        t_u_inv = (1/sqrt_u)*sqrt_Years_no_EQ
        t_inv_u = 1/t_u_inv
        a_inv = 1/sample_prop[:N]
        alpha_exp_fact = np.exp(2*np.square(a_inv))
        u1_t = a_inv*(t_u_inv-t_inv_u)
        u2_t = a_inv*(t_u_inv+t_inv_u)
        t1_u_inv = (1/sqrt_u)*sqrt_Years_no_EQ_plus_1
        t1_inv_u = 1/t1_u_inv
        u1_t1 = a_inv*(t1_u_inv-t1_inv_u)
        u2_t1 = a_inv*(t1_u_inv+t1_inv_u)
        p_T = norm.cdf(u1_t) + alpha_exp_fact*norm.cdf(-u2_t)
        p_T1 = norm.cdf(u1_t1) + alpha_exp_fact*norm.cdf(-u2_t1) 
        p_marg = (p_T1 - p_T)/(1-p_T)
        p_marg_inv = norm.ppf(p_marg)
        # Find Covariance matrix
        #sys.exit(1)
        corr_vector = np.exp(-sq_dist_low_sq_m/sample_prop[2*N]**2)   # Input according to 
        # (http://www.math.wsu.edu/faculty/genz/software/fort77/mvndstpack.f)
        # Interval
         # If there is rupture (e.g X(j)=1), the z < inv_Phi(p), then inf = 0 
            # (e.g. [-inf, inv_Phi(p)])
        infin = 1 - Rupture_history
        #log_lik
        log_lik_data = 0
        for j in range(n_years):
            #print j
            error,value,inform = mvn.mvndst(p_marg_inv[j,:],p_marg_inv[j,:],infin[j,:],corr_vector)
            log_lik_data += np.log(value)
      
        ####### Posterior_lik ######
        post_log_like = log_prior + log_lik_data
        A = np.exp(post_log_like - prev_post_log_lik)
        #print 'Acceptance Rate: ', A
        U = np.random.random_sample()
        if U < min(A,1):
            print "Moved"
            parameter_samples[i,:] = sample_prop
            prev_post_log_lik = post_log_like
            A_rate += 1
            parameter_likelihood[i] = post_log_like
        else:
            print "Did not move"
            parameter_samples[i,:] = parameter_samples[i-1,:]
            parameter_likelihood[i] = prev_post_log_lik
    else:
        print "Did not move"
        parameter_samples[i,:] = parameter_samples[i-1,:]
        parameter_likelihood[i] = prev_post_log_lik 

        
        
A_rate = float(A_rate)/M
    

########## Plot chain mixing ###########3
fig = plt.figure()
plt.scatter(parameter_samples[0,N+3-1],parameter_samples[0,N+4-1],color='red')
plt.scatter(parameter_samples[-1,N+3-1],parameter_samples[-1,N+4-1],color='black')
plt.plot(parameter_samples[:,N+3-1],parameter_samples[:,N+4-1])
plt.savefig('mixing.png')
plt.close()        


########## Plot prior and posterior distribution ###########3
fig = plt.figure()
inter_time = np.linspace(1,500,100)
plt.plot(inter_time,lognorm.pdf(inter_time, s=1,scale=median_parameter_prior[N+4-1]))
plt.hist(parameter_samples[:,N+4-1],normed=True)        
plt.savefig('prior_post.png')
plt.close()        

# Save variables
f = open(filename+'MCMC_Parameters_no_par', 'wb')
pickle.dump(A_rate, f)
pickle.dump(parameter_samples, f)
pickle.dump(parameter_likelihood, f)
pickle.dump(median_parameter_prior, f)
pickle.dump(sigma_walk,f)
pickle.dump(s, f)
f.close()





