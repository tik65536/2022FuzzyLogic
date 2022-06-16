#!/usr/bin/python3
import numpy as np
import time


def minmax(x,y):
    z = np.max(np.minimum(x,y),axis=1)
    return z.reshape(1,-1)

class DE_FRE():
    def __init__(self, initSize=200,maxiter=100,error=1e-03,R=None,b=None):
        self.best=[]
        self.mean=[]
        self.error = error
        self.pplSize = initSize
        self.maxiter = maxiter
        self.R=R
        self.b=b.reshape(1,-1)
        self.population = np.random.random((self.pplSize,self.R.shape[1]))
        #if torch.cuda.is_available():
        #    self.device = torch.device('cuda')
        #else:

    def fit(self,config,id_):
        config = config.reshape(1,-1)
        config = np.repeat(config,self.R.shape[0],axis=0)
        c = minmax(config,self.R)
        return np.sum((self.b-c)**2)

    def mutation_rand_1_z(self,x0,x1,x2,beta):
        # number of hidden layer mutation
        #[0] is H , [1] is W
        return x0+beta*(x1-x2)

    def crossoverRandomSwap(self,parent,u):
        # the first one is with min len
        swap = np.random.choice(2, self.R.shape[1])
        for i,s in enumerate(swap):
            if(s!=0):
               parent[i]=u[i]
        return parent

    def run(self,beta=0.5):
        current_gen=self.population
        scores = np.zeros((self.pplSize))
        print('DE Initial Run Start')
        for i in range(len(current_gen)):
            s = self.fit(current_gen[i],self.R)
            scores[i]=s
        print('DE Initial Run End')
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        bestGen = 0
        print(f'DE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}, config: {current_gen[currentbestidx]}')
        #Generation Run
        for i in range(self.maxiter):
            structureStatistic=np.zeros((self.pplSize,4))
            updatecount=0
            start=time.time()
            print(f'DE Gen {i} Run Start')
            for j in range(self.pplSize):
                parent = current_gen[j]
                idx0,idx1,idxt = np.random.choice(range(0,self.pplSize),3,replace=False)
                unitvector = self.mutation_rand_1_z(current_gen[idxt], current_gen[idx0], current_gen[idx1], beta)
                nextGen = self.crossoverRandomSwap(parent,unitvector)
                #print(f'DE Next Gen: {nextGen}')
                structureStatistic[j,0]= np.mean(nextGen)
                structureStatistic[j,1]= np.median(nextGen)
                structureStatistic[j,2]= np.quantile(nextGen,0.25)
                structureStatistic[j,3]= np.quantile(nextGen,0.75)
                s = self.fit(nextGen,self.R)
                if(s<scores[j]):
                    updatecount+=1
                    scores[j]=s
                    current_gen[j]=nextGen
            print(f'DE Gen {i} Run End')
            end=time.time()
            currentbest = np.min(scores)
            currentmean = np.mean(scores)
            currentmedian = np.median(scores)
            currentq25 = np.quantile(scores,0.25)
            currentq75 = np.quantile(scores,0.75)
            currentbestidx = np.argmin(scores)
            genMeanNode=np.median(structureStatistic[:,0])
            genMedianNode=np.median(structureStatistic[:,1])
            genq25Node = np.median(structureStatistic[:,2])
            genq75Node = np.median(structureStatistic[:,3])
            if(currentbest<overallBest):
                overallBest=currentbest
                overallBestConfig = current_gen[currentbestidx]
                bestGen = i
            print(f'DE Run {i:3d} CurrentBest: {currentbest:10.8f}, Mean: {currentmean:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, updatecount: {updatecount:3d}, Generation RunTime: {(end-start):10.8f}')
            print(f'Statistic| genMeanMedian : {genMeanNode:5.4f} | genMedianMedian  :{genMedianNode:5.4f} | gen25q : {currentq25:5.4f} | gen27q : {currentq75:5.4f} | updatecount : {updatecount:3d}' )
        print(f'DE Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}')
        return
R=[[0.56,0.94,0.33,0.61,0.98,1,0.67,0.83,0.11,0.17,0.72,0.22,0.23,0.28,0.44,0.31,0.78,0.89,0.5,0.39],
[0.13,0.56,0.22,0.89,0.33,0.94,0.78,0.39,0.16,0.26,0.28,0.11,0.17,0.18,1,0.3,0.44,0.67,0.42,0.2],
[0.22,0.24,0.44,0.38,0.67,0.94,0.56,0.48,0.27,0.28,0.29,0.17,0.11,0.3,0.72,0.32,0.41,1,0.61,0.5],
[0.44,0.67,0.11,0.22,0.16,0.31,0.24,0.56,0.27,0.89,1,0.2,0.29,0.33,0.72,0.31,0.37,0.78,0.41,0.32],
[0.22,0.23,0.56,0.26,0.44,0.28,0.17,0.67,0.33,0.89,0.11,0.78,0.7,1,0.3,0.61,0.76,0.31,0.39,0.32]]
R=np.array(R)
b=np.array([0.1853,0.8533,0.8016,0.8696,0.4412])
de = DE_FRE(R=R,b=b)
de.run()
