from AlignmentViaErGOT import *
import random
import time

GraphSize = [20,30,40,50,60,70,80,90,100,110,120]
RepeatNum = 20
RandSeed = np.arange(1,RepeatNum+1)

ErGot_Sigma1= np.zeros((len(GraphSize),RepeatNum,2))
ErGot_Sigma2= np.zeros((len(GraphSize),RepeatNum,2))
ErGot_Sigma3= np.zeros((len(GraphSize),RepeatNum,2))
ErGot_Sigma4= np.zeros((len(GraphSize),RepeatNum,2))

for ID in range(len(GraphSize)):
    n = GraphSize[ID]
    blocks = [int(n/2),int(n/2)]
    probs = [[0.7,0.05],[0.05,0.7]]
    for IDR in range(RepeatNum):
        print('Graph Size: %f - Iteration: %f ' % (n, IDR))
        [sbm_x, sbm_y, sbm_P_true] = GraphGeneration(n, blocks, probs, graph_type = 'sbm', seed_nb = RandSeed[IDR].item())
        ## ErGOT
        Gx, Gy, P_predict, TotalTimeCost = Optimization_ErGOT(sbm_x, sbm_y, IterNum=10, tau=2, Tmax=1000, epochs=1000, lr=0.5, epsilon=2000, alpha = 0.2, ones = 1, PlotLoss=False, Trick= True)                           
        L2Distance = Evaluation(sbm_x, sbm_y, P_predict)
        ErGot_Sigma1[ID,IDR,0]=L2Distance ## L2 distance
        ErGot_Sigma1[ID,IDR,1]=TotalTimeCost ## Time cost for running

        Gx, Gy, P_predict, TotalTimeCost = Optimization_ErGOT(sbm_x, sbm_y, IterNum=10, tau=2, Tmax=1000, epochs=1000, lr=0.5, epsilon=2000, alpha = 0.2, ones = 2, PlotLoss=False, Trick= True)                                 
        L2Distance = Evaluation(sbm_x, sbm_y, P_predict)
        ErGot_Sigma2[ID,IDR,0]=L2Distance ## L2 distance
        ErGot_Sigma2[ID,IDR,1]=TotalTimeCost ## Time cost for running

        Gx, Gy, P_predict, TotalTimeCost = Optimization_ErGOT(sbm_x, sbm_y, IterNum=10, tau=2, Tmax=1000, epochs=1000, lr=0.5, epsilon=2000, alpha = 0.2, ones = 3, PlotLoss=False, Trick= True)                                 
        L2Distance = Evaluation(sbm_x, sbm_y, P_predict)
        ErGot_Sigma3[ID,IDR,0]=L2Distance ## L2 distance
        ErGot_Sigma3[ID,IDR,1]=TotalTimeCost ## Time cost for running

        Gx, Gy, P_predict, TotalTimeCost = Optimization_ErGOT(sbm_x, sbm_y, IterNum=10, tau=2, Tmax=1000, epochs=1000, lr=0.5, epsilon=2000, alpha = 0.2, ones = 4, PlotLoss=False, Trick= True)                                 
        L2Distance = Evaluation(sbm_x, sbm_y, P_predict)
        ErGot_Sigma4[ID,IDR,0]=L2Distance ## L2 distance
        ErGot_Sigma4[ID,IDR,1]=TotalTimeCost ## Time cost for running

results = {'ErGOT L^{-1}':ErGot_Sigma1, 
            'ErGOT L':ErGot_Sigma2, 
            'ErGOT \exp(-2\alpha L)':ErGot_Sigma3, 
            'ErGOT L^2':ErGot_Sigma4, 
            }
np.save(r'D:\Files\LearningTheory\OptimalTransport\SIAM\NewResults\exp_results-sbm-trick-ErGOT.npy', results)