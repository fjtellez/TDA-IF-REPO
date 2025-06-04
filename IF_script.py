# numpy
import numpy as np
from numpy import asarray

# matplotlib
import matplotlib.pyplot as plt

# Persim for persistence landscapes
import persim
from persim import PersLandscapeExact
from persim.landscapes import plot_landscape_simple

# Pandas for working with excel
import pandas as pd

# Cubical homology calculator
import tcripser as tcr
import cripser as cr

# For handling images and paths
import os
from PIL import Image

import scipy
from scipy import integrate

from xlwt import Workbook

# For time tests
import time

# Colormaps
import matplotlib.cm as cm

# sklearn for clustering and silhouette analysis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# Function that computes number of loops in an image.
def n_loops(img):
    def f(x,y,theta=0):
        p=np.array([x,y])
        v=np.array([np.cos(theta),np.sin(theta)])
        return np.sum(p*v)
        
    x_length=len(img[:,0])
    y_length=len(img[0,:])
    
    # Filtering and processing to apply modified filtration
    gbm_filtered=np.zeros([x_length,y_length])

    for k in range(len(gbm_filtered[:,0])):
        for p in range(len(gbm_filtered[0,:])):
            if img[k,p]==1:
                gbm_filtered[k,p]=f(k,p,0)
            else:
                gbm_filtered[k,p]=700

    # compute PH for the V-construction of the original image (pixel value filtration)
    pd = cr.computePH(gbm_filtered)
    pds = [pd[pd[:,0] == i] for i in range(3)]
    diagrams=[pds[0][:-1,1:3],pds[1][:,1:3]]  

    return len(diagrams[1])


# Takes an image, centers it (between 0 and 1) and then computes the lateral and vertical limits of the connected components
# S[0] is direction [1,0], S[1] is direction [0,1], S[2] is direction [-1,0] and S[3] is direction [0,-1]
def filt_1d(img):

    Q=[np.array([1,0]),np.array([0,1]),np.array([-1,0]),np.array([0,-1])]

    n_img=np.zeros([len(img[:,0]),len(img[0,:])])
    n_img=img
    n_img[0,:]=1
    n_img[-1,:]=1
    n_img[:,0]=1
    n_img[:,-3]=1
    

    def f(x,y,p):
        return np.dot(p,np.array([x,y]))

    S=np.zeros(4)

    l=np.max([len(n_img[:,0]),len(n_img[0,:])])

    for k in range(len(Q)):
    
        gbm_filtered=np.zeros([len(n_img[:,0]),len(n_img[0,:])])

        R=1000

        for i in range(len(n_img[:,0])):
            for j in range(len(n_img[0,:])):
                if n_img[i,j]==1:
                    gbm_filtered[i,j]=f((i-len(n_img[:,0])/2)/l,(j-len(n_img[0,:])/2)/l,Q[k])
                else:
                    gbm_filtered[i,j]=R

        pd = cr.computePH(gbm_filtered)
        pds = [pd[pd[:,0] == i] for i in range(3)]
        P=[pds[0][:-1,1:3],pds[1][:,1:3]]

        ple=PersLandscapeExact(dgms=P,hom_deg=0)
        Landscape_points=ple.critical_pairs

        s=Landscape_points[0][0][1]
        for i in range(len(Landscape_points[0])):
            if Landscape_points[0][i][1]>s:
                s=Landscape_points[0][i][1]

        if k==2 or k==3:
            S[k]=-(R-2*s)
        else:
            S[k]=R-2*s

    n_img[0,:]=0
    n_img[-1,:]=0
    n_img[:,0]=0
    n_img[:,-1]=0
        
    return S

# Computes the value of the interior surface at a point
def interior_function(x,y,img,n_loops_img):

    def g(a,b):
        return -np.sqrt((a-x)**2+(b-y)**2)

    gbm_filtered=np.zeros([len(img[:,0]),len(img[0,:])])

    l=np.max([len(img[:,0]),len(img[0,:])])

    for i in range(len(img[:,0])):
        for j in range(len(img[0,:])):
            if img[i,j]==1:
                gbm_filtered[i,j]=g(i/(l-1),j/(l-1))
            else:
                gbm_filtered[i,j]=np.inf

    pd = cr.computePH(gbm_filtered)
    pds = [pd[pd[:,0] == i] for i in range(3)]
    P=[pds[0][:-1,1:3],pds[1][:,1:3]]

    s=0

    if len(P[1])>=1:
        ple=PersLandscapeExact(dgms=P,hom_deg=1)
        Landscape_points=ple.critical_pairs
        for i in range(len(Landscape_points)):
            for j in range(len(Landscape_points[i])):
                if Landscape_points[i][j][1]>s and Landscape_points[i][j][1]<np.inf:
                    s=Landscape_points[i][j][1]

    return 2*s

# For computing the integrals that are in the formula of necrotic lacunarity
def integral_img(img,error=2):
    S=filt_1d(img)
    n_loops_img=n_loops(img)
    return integrate.dblquad(lambda y,x: interior_function(x,y,img,n_loops_img),S[0],S[2],S[1],S[3],epsrel=error)[0]

