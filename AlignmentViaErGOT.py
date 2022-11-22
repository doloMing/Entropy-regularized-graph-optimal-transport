import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators import stochastic_block_model

import torch
torch.set_default_tensor_type('torch.DoubleTensor')
import random
import ot

import numpy.linalg as lg
import scipy.linalg as slg
import sklearn
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot
import time

import copy

import warnings
warnings.filterwarnings('ignore')

def Permutation(n, l1, seed_nb):
    np.random.seed(seed_nb)
    idx = np.random.permutation(n)
    P_true = np.eye(n)
    P_true = P_true[idx, :]
    l2 = np.array(P_true @ l1 @ P_true.T)
    
    return np.double(l2), P_true

def GraphGeneration(n,  block_sizes = [], block_prob = [], graph_type = 'er', seed_nb = 123):
        if graph_type == 'geo':
            g1 = nx.random_geometric_graph(n, 0.55)
        if graph_type == 'er':
            g1 = nx.erdos_renyi_graph(n, 0.45)
        if graph_type == 'sbm':
            g1 = stochastic_block_model(block_sizes, block_prob, seed = seed_nb)
        g1.remove_nodes_from(list(nx.isolates(g1)))
        n = len(g1)
        l1 = nx.laplacian_matrix(g1,range(n))
        l1 = np.array(l1.todense())
        # Permutation and second graph
        l2, P_true = Permutation(n, l1, seed_nb)
        x = np.double(l1)
        y = l2
        return [x, y, P_true]

def GraphRepresentation(x, alpha, ones, Trick):
    if Trick== False:
        if ones==1: # Sigma = L^(-1)+1/n J
            Gx = lg.pinv(x)+np.ones([len(x),len(x)])/len(x)
        elif ones==2: # Sigma = L+1/n J
            Gx = x +np.ones([len(x),len(x)])/len(x)
        elif ones==3: # Sigma = (exp(-alpha*L))^2
            Gx = lg.matrix_power(slg.expm(-alpha*x), 2)   
        elif ones==4: # Sigma = L^2
            Gx = lg.matrix_power(x, 2)  
    elif Trick== True:
        if ones==1: # Sigma = L^(-1)+1/n J
            Gx = lg.inv(x + 0.1*np.eye(len(x))+np.ones([len(x),len(x)])/len(x)) 
        elif ones==2: # Sigma = L+1/n J
            Gx = x +np.ones([len(x),len(x)])/len(x)+ 0.1*np.eye(len(x))
        elif ones==3: # Sigma = (exp(-alpha*L))^2
            Gx = lg.matrix_power(slg.expm(-alpha*x), 2)+ 0.1*np.eye(len(x))   
        elif ones==4: # Sigma = L^2
            Gx = lg.matrix_power(x, 2)+ 0.1*np.eye(len(x))
    return Gx


def DoublyStochasticMatrix(P, tau, IterNum):
    A = P / tau
    for i in range(IterNum):
        A = A - A.logsumexp(dim=1, keepdim=True)
        A = A - A.logsumexp(dim=0, keepdim=True)
    return torch.exp(A)

def SinkhornDiv(DS, x, y, SumEigMexx, LogProdEigMexx, epsilon):
    yy = torch.transpose(DS,0,1) @ y @ DS
    Mexy = 2*torch.eye(len(x))+16/(epsilon**2)*torch.mm(x,yy)
    Meyy = 2*torch.eye(len(yy))+16/(epsilon**2)*torch.mm(yy,yy)
    Term1 = SumEigMexx + torch.trace(-2*Mexy+Meyy)  
    #Term2 = 2*torch.log(torch.linalg.det(Mexy))-LogProdEigMexx-torch.log(torch.linalg.det(Meyy)) （Either expression is fine）
    Term2 = 2*torch.trace(torch.log(Mexy))-LogProdEigMexx-torch.trace(torch.log(Meyy)) # （Either expression is fine）
    cost = epsilon/4*(Term1+Term2)
    return cost


def ErGOT(Gx, Gy, IterNum, tau, Tmax, epochs, lr, epsilon, PlotLoss=True, verbose=True):
    x = torch.from_numpy(Gx.astype(np.double))
    y = torch.from_numpy(Gy.astype(np.double))
    
    Nx = x.shape[0]
    L = torch.randn(Nx, Nx, requires_grad=True)

    Mexx = 2*torch.eye(len(x))+16/(epsilon**2)*torch.mm(x,x)
    u, EigMexx, v = torch.svd(Mexx)
    SumEigMexx = torch.sum(EigMexx)
    LogProdEigMexx = torch.log(torch.prod(EigMexx))
    optimizer = torch.optim.Adam([L], lr=lr, amsgrad=True)
    Scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Tmax)
    history = []
    CurrentLr = []
    StartTime = time.time()
    TotalTimeCost = 0
    for epoch in range(epochs):
        P_noisy = L
        DS = DoublyStochasticMatrix(P_noisy, tau, IterNum)
        LossValue = SinkhornDiv(DS, x, y, SumEigMexx, LogProdEigMexx, epsilon)
        cost = LossValue
        # Gradient step
        cost.backward()
        optimizer.step()
        Scheduler.step()
        optimizer.zero_grad()
        # Tracking
        history.append(cost.item())
        Cur_Lr=optimizer.param_groups[-1]['lr']
        CurrentLr.append(Cur_Lr)
        if verbose and (epoch==0 or (epoch+1) % 100 == 0):
            CurrentTime=time.time()
            print('[Epoch %4d/%d] loss: %f - Lr: %f - time cost: %s' % (epoch+1, epochs, cost.item(), Cur_Lr, CurrentTime-StartTime))
            TotalTimeCost=TotalTimeCost+CurrentTime-StartTime
            StartTime=time.time()
    # PyTorch -> NumPy
    P = DoublyStochasticMatrix(L, tau, IterNum)
    P = P.squeeze()
    P = P.detach().numpy()
    # Keep the max along the rows
    idx = P.argmax(1)
    P = np.zeros_like(P)
    P[range(Nx),idx] = 1.
    # Convergence plot
    if PlotLoss == True:
        plt.figure(figsize=(15,10))
        plt.subplot(2,2,1)
        plt.plot(history)
        plt.title("Loss")
    return P,TotalTimeCost 

def Optimization_ErGOT(x, y, IterNum, tau, Tmax, epochs, lr, epsilon, alpha = 2, ones = 1, PlotLoss=True, Trick=False):
    Gx = GraphRepresentation(x, alpha, ones,Trick)
    Gy = GraphRepresentation(y, alpha, ones,Trick)
    P, TotalTimeCost = ErGOT(Gx, Gy, IterNum, tau, Tmax, epochs, lr, epsilon, PlotLoss=True, verbose=True)
    return Gx, Gy, P, TotalTimeCost 

def Evaluation(x, y, P):
    L2Distance = lg.norm(x -  P.T @ y @ P, 'fro')
    return L2Distance