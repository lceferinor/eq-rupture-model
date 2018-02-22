"""
%% Bayesian parameter estimation for rupture occurrence model: (Ceferino et al. BSSA (2018))
% Developed by: Luis Ceferino
% Date: 02/21/2017
-----------------------------------------------------------------------------
% Script to run Bayesian inference using MCMC using MPI parallel processing
% Each certain number of iterations, the processors exchange the most likely 
% samples at the time
-----------------------------------------------------------------------------
"""
# Required imports
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.stats import mvn
from scipy.stats import norm
import pickle
import timeit


start_t = timeit.default_timer()
filename = 'Rupture_data_hitoric'
##################### Parallel Settings ######################3
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = MPI.COMM_WORLD.Get_size()
print size
size = 4
#print size
np.random.seed(rank)
# Defining number of samples and distribution among processors
M_total = 2**12 # around 4000 samples
n_exchange = 2**8
running_len = M_total/n_exchange
M_total = 128 # around 4000 samples
running_len = 4
M = int(M_total/(size)) # Number of samples per processor


###################### DATA #############################
# Load variables
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

B = 0 # Burning period
T_p = 2*N+1 # Number of parameters

parameter_samples = np.zeros((M,T_p)) # Column: N aplha parameters, N mu parameters, and gamma
    									  # Row: M samples
parameter_likelihood = np.zeros(M) # Column: N aplha parameters, N mu parameters, and gamma
    									  # Row: M samples
parameter_samples[0,:] = np.array([1.0]*N + [150.0]*N + [300.0]) #Initial value

######################## Define random walk in MCMC ###################
sigma_walk = np.array([.04]*N + [4.0]*N + [7.0])*5
######################## Prior ##################################
median_parameter_prior = np.array([.92, 1.22, 0.68, .73, .99, 1.08, 0.75, .96,\
                                    320.0, 172.0, 194.0, 97.0, 114.0, 110.0, 144.0, 146.0, 290.0]) 
                            # From the previous MLE analysis (ICOSSAR)
s = 0.3 # Same log stand dev for all
#log_std_parameter_prior = np.array([1]*N + [1]*N + [1])


################# Compute MCMC ##########################
prev_post_log_lik = -float('inf') # Always accept initial sample
Acc = 0


for i in range(1,M):
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
            error,value,inform = mvn.mvndst(p_marg_inv[j,:],p_marg_inv[j,:],infin[j,:],corr_vector)
            log_lik_data += np.log(value)
      
        ####### Posterior_lik ######
        post_log_like = log_prior + log_lik_data
        A = np.exp(post_log_like - prev_post_log_lik)
        #print 'Acceptance Rate: ', A
        U = np.random.random_sample()
        if U < min(A,1):
            print "Sample: ", i, ' in process ', rank, ': Moved'
            parameter_samples[i,:] = sample_prop
            prev_post_log_lik = post_log_like
            Acc += 1
            parameter_likelihood[i] = post_log_like
        else:
            print "Sample: ", i, ' in process ', rank, ': Did not move'
            parameter_samples[i,:] = parameter_samples[i-1,:]
            parameter_likelihood[i] = post_log_like
    else:
        print "Sample: ", i, ' in process ', rank, ': Did not move'
        parameter_samples[i,:] = parameter_samples[i-1,:]
        parameter_likelihood[i] = post_log_like
    # Choose selected ones

	# Exchange the most likely samples among the processors
    likelihood_processors = np.zeros(size)
    points_processors = np.zeros((size,T_p))
    if i%running_len == 0:
        
        
        if rank == 0:
            
            likelihood_processors[0] = post_log_like
            points_processors[0,:] = parameter_samples[i,:]
            recv_buffer_post_log_lik = np.zeros(1)
            recv_buffer_point = np.zeros(T_p)
            for proc in range(1, size):
                #print 'hola bb'
                comm.Recv(recv_buffer_post_log_lik, source = proc)
                comm.Recv(recv_buffer_point, source = proc)
                likelihood_processors[proc] = recv_buffer_post_log_lik[0]
                points_processors[proc,:] = recv_buffer_point
            #print 'Proc 0 received'
            for proc in range(1, size):
                comm.Send(likelihood_processors,dest = proc)
                comm.Send(points_processors,dest = proc)
            
                
        else:
            # Send likelihood and point
            send_buffer_post_lik = np.zeros(1)
            send_buffer_post_lik[0] = parameter_likelihood[i]
            send_buffer_points = parameter_samples[i,:]
            comm.Send(send_buffer_post_lik,dest = 0)
            comm.Send(send_buffer_points,dest = 0)
            #print 'Sent by proc: ', rank
            # Receive status of other processors
            recv_buffer_post_liks = np.zeros(size)
            recv_buffer_points = np.zeros((size,T_p))
            comm.Recv(recv_buffer_post_liks, source = 0)
            comm.Recv(recv_buffer_points, source = 0)
            likelihood_processors = recv_buffer_post_liks
            points_processors = recv_buffer_points
                        
        #Update statuses
        proc_ord = np.argsort(likelihood_processors)
        index_proc = np.where(proc_ord == rank)[0][0]      
        if index_proc < size/2.0:
            parameter_samples[i,:] = points_processors[proc_ord[int(index_proc + size/2)],:]
            parameter_likelihood[i] = likelihood_processors[proc_ord[int(index_proc + size/2)]]

            

##### Rank 0 is master processor
if rank == 0:
    parameter_samples_all = parameter_samples[B:,:]
    parameter_likelihood_all = parameter_likelihood[B:]
    recv_buffer_param = np.zeros(parameter_samples.shape)
    recv_buffer_lik = np.zeros(parameter_likelihood.shape)
    recv_buffer_A = np.zeros(1)
    Acc_total = Acc
	# Receive data from other processors
    for proc in range(1, size):
        comm.Recv(recv_buffer_param, source = proc)
        comm.Recv(recv_buffer_lik, source = proc)
        comm.Recv(recv_buffer_A, source = proc)
        parameter_samples_all = np.append(parameter_samples_all, 
                                          recv_buffer_param, axis=0)
        parameter_likelihood_all = np.append(parameter_likelihood_all, 
                                          recv_buffer_lik, axis=0)
        Acc_total += recv_buffer_A[0]

        
    A_rate = float(Acc_total)/(size*M)
    print 'The rate is: ', A_rate
    
        
    
    ########## Plot prior and posterior distribution ###########3
    fig = plt.figure()
    inter_time = np.linspace(1,500,100)
    plt.plot(inter_time,lognorm.pdf(inter_time, s=1,scale=median_parameter_prior[N+4-1]))
    plt.hist(parameter_samples_all[:,N+4-1],normed=True)        
    plt.savefig('prior_post.png')
    plt.close()        
    


    # Save variables
    f = open(filename+'_MCMC_Parameters_parallel', 'wb')
    pickle.dump(A_rate, f)
    pickle.dump(parameter_samples_all, f)
    pickle.dump(parameter_likelihood_all, f)
    pickle.dump(median_parameter_prior, f)
    pickle.dump(sigma_walk,f)
    pickle.dump(s, f)
    f.close()
    end_t = timeit.default_timer()
    print 'The analysis with took ', end_t - start_t, ' seconds'
    

# Other processors will send data to rank 0        
else:
    send_buffer_parameter = parameter_samples[B:,:]
    send_buffer_A = np.zeros(1)
    send_buffer_lik = parameter_likelihood[B:]
    send_buffer_A[0] = Acc
    comm.Send(send_buffer_parameter,dest = 0)
    comm.Send(send_buffer_lik,dest = 0)
    comm.Send(send_buffer_A,dest = 0)
       

