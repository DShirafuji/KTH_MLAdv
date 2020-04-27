import pylab as pb
import numpy as np
from math import pi
from scipy.spatial.distance import  cdist
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import random


def plot2dGaussian(mu,cov,w0,w1,title):
	dist=multivariate_normal(mu,cov)
	X,Y=np.meshgrid(w0,w1)
	Z = np.zeros((len(w0),len(w1)))
	for i in range(len(w0)):
		for j in range(len(w1)):
			Z[i,j] = multivariate_normal(mu, cov).pdf([X[i,j],Y[i,j]])
	plt.contourf(X,Y,Z,cmap = 'jet')
	plt.xlabel('w_1')
	plt.ylabel('w_0')
	plt.title(title)
	plt.axis('scaled')
	plt.show()



def calPosterior(prior_mu,prior_cov,noise_mu,noise_cov,X_Label,T,points):
	X_observed=np.ones(points)
	T_observed=np.ones(points)
	for i in range(points):
		temp=random.randint(0,X_Label.shape[0]-1)
		X_observed[i]=X_Label[temp,1]
		T_observed[i]=T[temp,0]
		# print(temp)
	X_observed=X_observed.reshape(-1,1)
	X_observed=np.hstack((X_observed,np.ones(X_observed.shape[0]).reshape(-1,1)))
	X_observed=np.flip(X_observed,axis=1)
	T_observed=T_observed.reshape(-1,1)
	posterior_cov=np.linalg.inv((1./(noise_cov))*(np.matmul(X_observed.transpose(),X_observed))+np.linalg.inv(prior_cov))
	posterior_mean=np.matmul(posterior_cov,np.matmul(X_observed.transpose(), T_observed))/noise_cov
	posterior_mean=posterior_mean.flatten()
	return posterior_mean,posterior_cov


# Create Data
W=np.array([-1.5,0.5]).reshape(-1,1)
X_Label=np.arange(-1,1.01,.01)
X_Label=np.stack((X_Label,np.ones(len(X_Label))),axis = 1)
X_Label=np.flip(X_Label,axis=1)
noise_mu=0
noise_cov=0.2
noise = np.random.normal(noise_mu, noise_cov, X_Label.shape[0])
noise=noise.reshape(-1,1)
T=np.matmul(X_Label,W)+noise


# Set Sigma <- CHANGE HERE like: 0.1, 0.2, 0.4, 0.8
sigma = 0.8
# plot prior
prior_mu=np.array([0,0])
prior_cov=np.array([[sigma,0],[0,sigma]])
w0=np.arange(-2,2,sigma)
w1=np.arange(-2,2,sigma)
title=''
plot2dGaussian(prior_mu,prior_cov,w0,w1,title)

# How many data points???
# : N
N=[1,3,5,7]
for n in N:
	posterior_mean,posterior_cov=calPosterior(prior_mu,prior_cov,noise_mu,noise_cov,X_Label,T,n)
	w0=np.arange(-2,2,sigma)
	w1=np.arange(-2,2,sigma)
	title=n
	plot2dGaussian(posterior_mean,posterior_cov,w0,w1,title)
	# draw 5 samples from posterior and plot function
    # We plot 5 samples now
	for i in range(5):
		w_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, 5)
	p=np.arange(-2,2,.5)
	plt.plot(p, 0.5*p-1.5,label="True", color = "blue")
	for i in range(5):
		# Plot the line based on samples
		plt.plot(p, w_samples[i][1]*p + w_samples[i][0], label="Samples", color = "red", linestyle = "-", linewidth = 0.5)
	plt.legend()
	plt.show()
