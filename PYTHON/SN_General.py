import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal as mvt
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import bernoulli
import scipy.integrate as integrate
from scipy.stats import kstest
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import direct
from scipy.linalg import block_diag
import seaborn as sns
from itertools import combinations_with_replacement
from itertools import product
import os
import pickle
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import scipy

def Features(data, d):
  '''
  Function to create a DataFrame of features from data
  '''

  cols = data.columns

  # List of monomials. The first entry corresponds to the constant. We conly consider the non-constant elements.
  coeffs = [i for i in product(range(d+1),repeat = len(cols)) if sum(i)<= d][1:]


  features = {}
  for coef in coeffs:
    column_name = str(coef)
    col_value = np.ones(data.shape[0])
    j = 0
    for i in coef:
      col_value = col_value * data[cols[j]]**i
      j+=1

    features[column_name] = col_value
  return pd.DataFrame(features)

def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

def get_c(mu, Sigma,d, low,high, Random_Feature_Sample):
  z = mvt.pdf(Random_Feature_Sample,mu,Sigma, allow_singular=True)
  vol = (high-low).prod()
  return [vol * np.mean(z)]

def get_Feature_Space_Estimates(data,d, useful_list = None):
  mat = Features(data, d)
  mat = pd.DataFrame(mat)
  if useful_list != None:
    mat = mat[useful_list]
  return (mat.mean(),mat.cov())

def Estimated_pdf(data,mu,Sigma,c,d,  useful_list = None):
  A = Features(data,d)
  if useful_list != None:
    A = A[useful_list]
  out = mvt.pdf(A,mu,Sigma,allow_singular=True)/c
  return out


def create_optim_vector(mu, Sigma):
  '''
   -   Vector used for optimization.
   -   First d elements of output is mu
   -   Next d(d+1)/2 elements are the elements of the upper triangular matrix cholesky(Sigma)
   -   It is a bijective function of (mu, Sigma)
   -   func 'get_mu_sigma_from_optim_vector' is the inverse of this function
  '''
  D = len(mu)
  A = np.linalg.cholesky(Sigma).T
  A = A[np.triu_indices(D,0)]
  return np.concatenate((mu,A))

def get_mu_sigma_from_optim_vector(x,d,D):
  '''
  Recovers (mu, Sigma) from optim vector
  '''
  mu = x[0:D]
  A = x[D:]
  upper = np.zeros((D, D))
  upper[np.triu_indices(D, 0)] = A
  Sigma = np.matmul(upper.T,upper)
  return (mu,Sigma)

def get_likelihood(Features,mu,Sigma,d,low,high ,Feature_Space_Samples,c_bound = 0.01):
  '''
  Returns log likelihood
  '''

  c = get_c(mu,Sigma,d,low,high,Feature_Space_Samples)[0]
  if c <= c_bound:
    c = c_bound

  Z = mvt.pdf(Features,mu,Sigma, allow_singular=True)
  Z[Z<(10**-22)] = 10**-22
  l = (np.log(Z/c)).sum()
  if np.random.random(1)<0.01:
    print('obj_value: '+str(l),' c: ' +str(c))
  return l


def func(x, Features,Feature_Space_Samples,d, D,low,high,c_bound = 0.01): # return - infty if c< c_bound
  '''
  Used for optimization
  '''
  mu, Sigma = get_mu_sigma_from_optim_vector(x,d,D)
  l = get_likelihood(Features,mu,Sigma,d, low,high,Feature_Space_Samples,c_bound)
  return l

  def get_heatmap(mu,Sigma, low = (-0.5,-0.2), high = (0.5,0.8),c = None, d = 2):
    if c==None:
      c = get_c(mu,Sigma)
    xaxis = np.linspace(low[0], high[0], 100)
    yaxis = np.linspace(low[1], high[1], 100)
    x,y = np.meshgrid(xaxis, yaxis)
    mesh = pd.DataFrame()
    mesh[0] = x.reshape(-1)
    mesh[1] = y.reshape(-1)
    mesh['pdf'] = Estimated_pdf(mesh,mu,Sigma,c,d)
    df_pivot = mesh.pivot(index=1, columns=0, values='pdf')

    df_pivot.index = pd.Series(df_pivot.index).round(2)
    df_pivot.columns = pd.Series(df_pivot.columns).round(2)
    sns.heatmap(df_pivot.iloc[::-1])

def get_Data_Space_Estimates(data,d, c_bound,low,high, niter = 20, ftol = 1e-2, tol = 1e-2, useful_list = None):
  # Initial Choice
  mu_0,Sigma_0 = get_Feature_Space_Estimates(data,d,useful_list )
  vect = create_optim_vector(mu_0, Sigma_0)

  features = Features(data,d)
  D = len(mu_0)
  A = np.random.uniform(low,high, size = (100000,len(low)))
  Feature_Space_Samples = Features(pd.DataFrame(A),d)


  if useful_list!=None:
    features = features[useful_list]
    Feature_Space_Samples = Feature_Space_Samples[useful_list]

  print('Initail Vector: ',vect)
  print('Initail obj Value : ', func(vect,features,Feature_Space_Samples,d,D, low, high))
  print('Initail c (Randomized): ', get_c(mu_0,Sigma_0,d,low,high,Feature_Space_Samples)[0])


  print('Optimizing \n')
  # Optimize
  minimizer_kwargs = {'method': 'L-BFGS-B', 'tol':tol,'options': {'ftol': ftol}}
  min = basinhopping(lambda x: -func(x,features,Feature_Space_Samples,d,D,low,high,c_bound), vect, niter = niter, minimizer_kwargs = minimizer_kwargs)

  # Recover parameters
  x = min['x']
  mu, Sigma = get_mu_sigma_from_optim_vector(x,d,D)

  return(mu,Sigma)



# Visualizations:

def get_heatmap(mu,Sigma, low = (-0.5,-0.2), high = (0.5,0.8),c = None, d = 2):
  if c==None:
    print('c is needed')
    return -1
  xaxis = np.linspace(low[0], high[0], 100)
  yaxis = np.linspace(low[1], high[1], 100)
  x,y = np.meshgrid(xaxis, yaxis)
  mesh = pd.DataFrame()
  mesh[0] = x.reshape(-1)
  mesh[1] = y.reshape(-1)
  mesh['pdf'] = Estimated_pdf(mesh,mu,Sigma,c,d)
  df_pivot = mesh.pivot(index=1, columns=0, values='pdf')

  df_pivot.index = pd.Series(df_pivot.index).round(2)
  df_pivot.columns = pd.Series(df_pivot.columns).round(2)
  sns.heatmap(df_pivot.iloc[::-1])


def get_contour(data, mu,Sigma, low = (-0.5,-0.2), high = (0.5,0.8),c = None, d = 2):
  '''
    - data must be bivariate for using this
    - d can be anything
    - INPUT: data, mu, Sigma. OPTIONAL:  low, high, c, d
    - OUTPUT: Contour plot of bivariate data with estimated pdf
  '''

  if c==None:
    c = get_c(mu,Sigma)
  xaxis = np.linspace(low[0], high[0], 100)
  yaxis = np.linspace(low[1], high[1], 100)
  x,y = np.meshgrid(xaxis, yaxis)
  mesh = pd.DataFrame()
  mesh[0] = x.reshape(-1)
  mesh[1] = y.reshape(-1)
  mesh['pdf'] = Estimated_pdf(mesh,mu,Sigma,c,d)

  shape = x.shape
  # Plot the scatter plot of the bivariate data
  plt.scatter(data.iloc[:,0], data.iloc[:,1], c='pink', s=1, label='Data points')

# Plot the contour lines of the estimated pdf
  X = np.array(mesh[0]).reshape(shape)
  Y = np.array(mesh[1]).reshape(shape)
  Z = np.array(mesh['pdf']).reshape(shape)
  contour = plt.contour(X,Y,Z, levels=4, cmap='viridis')
  plt.colorbar(contour, label='Estimated PDF')

  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  plt.title('Bivariate Data with Estimated PDF Contours')
  plt.legend()
  plt.show()

def get_contour_from_mesh(data, mesh, shape, title = 'Bivariate Data with Estimated PDF Contours', x = 'X-axis', y = 'Y-axis'):
  '''
    - Used when dimension of data > 2
    - Need to create a 'Mesh' separately
    - The Mesh is a dataframe with three columns   
      - First two columns represent points on a 2D space
      - Third column is the estimated bivariate marginal pdf
      - When data has more dimensions, the bivariate marginal is obtained by integrating (adding) out the other variables.
      - The input 'Mesh' is a product of this pre-processing to obtain the marginals.
    - shape: (len(xaxis),len(yaxis)) where xaxis and yaxis were used to create the mesh (grid)
  '''
  plt.scatter(data.iloc[:,0], data.iloc[:,1], c='pink', s=1, label='Data points')

# Plot the contour lines of the estimated pdf
  X = np.array(mesh.iloc[:,0]).reshape(shape)
  Y = np.array(mesh.iloc[:,1]).reshape(shape)
  Z = np.array(mesh['pdf']).reshape(shape)
  contour = plt.contour(X,Y,Z, levels=4, cmap='viridis')
  plt.colorbar(contour, label='Estimated PDF')

  plt.xlabel(x)
  plt.ylabel(y)
  plt.title(title)
  plt.legend()
  plt.show()
