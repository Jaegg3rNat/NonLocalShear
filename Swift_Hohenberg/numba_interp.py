#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:36:55 2021

@author: joao-valeriano
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit, set_num_threads
from tqdm import tqdm

set_num_threads(4)

@njit
def interp1d(x, xp, fp):
    fn = np.zeros(len(x))
    
    dx = np.diff(xp)[0]
    
    for i in range(len(x)):
        if x[i] <= xp[0]:
            fn[i] = fp[0]
        
        elif x[i] >= xp[-1]:
            fn[i] = fp[-1]

        else:
            ceil = int((x[i]-xp[0])/dx)+1
            
            fn[i] = fp[ceil-1] + (fp[ceil]-fp[ceil-1])/(xp[ceil]-xp[ceil-1])*(x[i]-xp[ceil-1])
                
    return fn

@njit
def interp1d_periodic(x, xp, fp):
    fn = np.zeros(len(x))
    x0 = 0
    
    for i in range(len(x)):
        x0 = 0
        xi = (x[i]-xp[0]) % (xp[-1]-xp[0])
        
        if xi <= xp[0]:
            fn[i] = fp[0]
        
        elif xi >= xp[-1]:
            fn[i] = fp[-1]

        else:
            while xp[x0] < xi:
                x0 += 1
    
            ceil = x0
            
            fn[i] = fp[ceil-1] + (fp[ceil]-fp[ceil-1])/(xp[ceil]-xp[ceil-1])*(xi-xp[ceil-1])
            
    return fn
        
# x = np.linspace(0, 2*np.pi, 30)
# y = np.sin(x)
# plt.plot(x, y, "o")
# x1 = np.linspace(-np.pi, 3*np.pi, 100)
# y_true = np.sin(x1)
# y1 = interp1d(x1, x, y)
# plt.plot(x1, y1)
# plt.plot(x1, interp1d_periodic(x1, x, y))
# # plt.plot(x1, y_true)
# plt.show()

@njit
def interp2d(x, y, xp, yp, fp, rescale=False):
    
    fn = np.zeros_like(x)
    
    # dx = x[:,1]-x[:,0]
    # dy = y[1,0]-y[0,0]
    dx = np.diff(xp)[0]
    dy = np.diff(yp)[0]
    
    if rescale:
        r = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            
            ceil_x = int((x[i,j]-xp[0])/dx)+1
            ceil_y = int((y[i,j]-yp[0])/dy)+1
            
            if x[i,j] <= xp[0]:
                
                if y[i,j] <= yp[0]:
                    fn[i,j] = fp[0,0]
                
                elif y[i,j] >= yp[-1]:
                    fn[i,j] = fp[0,-1]
                    
                else:    
                    fn[i,j] = fp[0, ceil_y-1] + (fp[0,ceil_y]-fp[0,ceil_y-1])/dy*(y[i,j]-yp[ceil_y-1])
            
            elif x[i,j] >= xp[-1]:
                
                if y[i,j] <= yp[0]:
                    fn[i,j] = fp[-1,0]
                
                elif y[i,j] >= yp[-1]:
                    fn[i,j] = fp[-1,-1]
                    
                else:
                    fn[i,j] = fp[-1, ceil_y-1] + (fp[-1,ceil_y]-fp[-1,ceil_y-1])/dy*(y[i,j]-yp[ceil_y-1])
            
            else:
                ceil_x = int((x[i,j]-xp[0])/dx)+1
                
                if y[i,j] <= yp[0]:                    
                    fn[i,j] = fp[ceil_x-1,0] + (fp[ceil_x,0]-fp[ceil_x-1,0])/dx*(x[i,j]-xp[ceil_x-1])
                
                elif y[i,j] >= yp[-1]:
                    fn[i,j] = fp[ceil_x-1,-1] + (fp[ceil_x,-1]-fp[ceil_x-1,-1])/dx*(x[i,j]-xp[ceil_x-1])
                    
                else:
                    ceil_y = int((y[i,j]-yp[0])/dy)+1
                    
                    fn[i,j] = (fp[ceil_x-1,ceil_y-1]*(xp[ceil_x]-x[i,j])*(yp[ceil_y]-y[i,j])
                              +fp[ceil_x,ceil_y-1]*(x[i,j]-xp[ceil_x-1])*(yp[ceil_y]-y[i,j]) 
                              +fp[ceil_x-1,ceil_y]*(xp[ceil_x]-x[i,j])*(y[i,j]-yp[ceil_y-1])
                              +fp[ceil_x,ceil_y]*(x[i,j]-xp[ceil_x-1])*(y[i,j]-yp[ceil_y-1]))/(dx*dy)
            
            if rescale:
                if xp[0] <= x[i,j] <= xp[-1] and yp[0] <= y[i,j] <= yp[-1]:
                    r[i,j] = fn[i,j]
                    fn[i,j] = 0.
    
    if rescale:
        fn += r*np.sum(fp)/np.sum(r)
    
    return fn

@njit
def interp2d_periodic_x(x, y, xp, yp, fp, rescale=False):
    
    fn = np.zeros_like(x)
    
    # dx = x[:,1]-x[:,0]
    # dy = y[1,0]-y[0,0]
    dx = np.diff(xp)[0]
    dy = np.diff(yp)[0]
    
    if rescale:
        r = np.zeros_like(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            
            if x[i,j] <= xp[0] or x[i,j] >= xp[-1]:
                x[i,j] = (x[i,j] - xp[0]) % (xp[-1]-xp[0]) + xp[0]
            
            ceil_x = int((x[i,j]-xp[0])/dx)+1
            ceil_y = int((y[i,j]-yp[0])/dy)+1
            
            if x[i,j] <= xp[0]:
                
                if y[i,j] <= yp[0]:
                    fn[i,j] = fp[0,0]
                
                elif y[i,j] >= yp[-1]:
                    fn[i,j] = fp[0,-1]
                    
                else:    
                    fn[i,j] = fp[0, ceil_y-1] + (fp[0,ceil_y]-fp[0,ceil_y-1])/dy*(y[i,j]-yp[ceil_y-1])
            
            elif x[i,j] >= xp[-1]:
                
                if y[i,j] <= yp[0]:
                    fn[i,j] = fp[-1,0]
                
                elif y[i,j] >= yp[-1]:
                    fn[i,j] = fp[-1,-1]
                    
                else:
                    fn[i,j] = fp[-1, ceil_y-1] + (fp[-1,ceil_y]-fp[-1,ceil_y-1])/dy*(y[i,j]-yp[ceil_y-1])
            
            else:
                ceil_x = int((x[i,j]-xp[0])/dx)+1
                
                if y[i,j] <= yp[0]:                    
                    fn[i,j] = fp[ceil_x-1,0] + (fp[ceil_x,0]-fp[ceil_x-1,0])/dx*(x[i,j]-xp[ceil_x-1])
                
                elif y[i,j] >= yp[-1]:
                    fn[i,j] = fp[ceil_x-1,-1] + (fp[ceil_x,-1]-fp[ceil_x-1,-1])/dx*(x[i,j]-xp[ceil_x-1])
                    
                else:
                    ceil_y = int((y[i,j]-yp[0])/dy)+1
                    
                    fn[i,j] = (fp[ceil_x-1,ceil_y-1]*(xp[ceil_x]-x[i,j])*(yp[ceil_y]-y[i,j])
                              +fp[ceil_x,ceil_y-1]*(x[i,j]-xp[ceil_x-1])*(yp[ceil_y]-y[i,j]) 
                              +fp[ceil_x-1,ceil_y]*(xp[ceil_x]-x[i,j])*(y[i,j]-yp[ceil_y-1])
                              +fp[ceil_x,ceil_y]*(x[i,j]-xp[ceil_x-1])*(y[i,j]-yp[ceil_y-1]))/(dx*dy)
            
            if rescale:
                if xp[0] <= x[i,j] <= xp[-1] and yp[0] <= y[i,j] <= yp[-1]:
                    r[i,j] = fn[i,j]
                    fn[i,j] = 0.
    
    if rescale:
        fn += r*np.sum(fp)/np.sum(r)
    
    return fn

@njit
def interp2d_periodic(x, y, xp, yp, fp, rescale=False):
    
    fn = np.zeros_like(x)
    
    # dx = x[:,1]-x[:,0]
    # dy = y[1,0]-y[0,0]
    dx = np.diff(xp)[0]
    dy = np.diff(yp)[0]
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            
            if x[i,j] <= xp[0] or x[i,j] >= xp[-1]:
                x[i,j] = (x[i,j] - xp[0]) % (xp[-1]-xp[0]) + xp[0]
            
            if y[i,j] < yp[0] or y[i,j] >= yp[-1]:
                y[i,j] = (y[i,j] - yp[0]) % (yp[-1]-yp[0]) + yp[0]

            ceil_x = int((x[i,j]-xp[0])/dx)+1
            ceil_y = int((y[i,j]-yp[0])/dy)+1
            
            fn[i,j] = (fp[ceil_x-1,ceil_y-1]*(xp[ceil_x]-x[i,j])*(yp[ceil_y]-y[i,j])
                      +fp[ceil_x,ceil_y-1]*(x[i,j]-xp[ceil_x-1])*(yp[ceil_y]-y[i,j]) 
                      +fp[ceil_x-1,ceil_y]*(xp[ceil_x]-x[i,j])*(y[i,j]-yp[ceil_y-1])
                      +fp[ceil_x,ceil_y]*(x[i,j]-xp[ceil_x-1])*(y[i,j]-yp[ceil_y-1]))/(dx*dy)

    if rescale:
        fn *= np.sum(fp)/np.sum(fn)
    
    return fn