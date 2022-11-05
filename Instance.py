from AlignmentViaErGOT import *
import random
import time

n = 50
blocks = [int(n/2),int(n/2)]
probs = [[0.7,0.05],[0.05,0.7]]
[x, y, P_true] = GraphGeneration(n, blocks, probs, graph_type = 'sbm', seed_nb = 50)
Gx, Gy, P_predict, TotalTimeCost = Optimization_ErGOT(x, y, it=10, tau=2, n_samples=10, epochs=1000, lr=0.5, epsilon=500, alpha = 0.2, ones = 1, PlotLoss=False, Trick=True)
L2Distance =Evaluation(x, y, P_predict)