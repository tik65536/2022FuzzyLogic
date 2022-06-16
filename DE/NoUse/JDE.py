#!/bin/python3
import numpy as np
import time

def minmax(x,y):
    z = np.max(np.minimum(x,y),axis=1)
    return z.reshape(1,-1)

class JDEV2_FRE():
    def __init__(self, initSize=200,maxiter=100,error=1e-03,R=None,b=None):
        self.best=[]
        self.mean=[]
        self.error = error
        self.pplSize = initSize
        self.maxiter = maxiter
        self.R=R
        self.b=b.reshape(1,-1)
        self.population = np.random.random((self.pplSize,self.R.shape[1]))
        self.adap_conf = (0.1,0.1,0.1,0.9)

    # define fit function - it calculates the fitness of one individual (one NN)
    def fit(self,config,id_):
        config = config.reshape(1,-1)
        config = np.repeat(config,self.R.shape[0],axis=0)
        c = minmax(config,self.R)
        return np.sum((self.b-c)**2)

    def jde_params(self,beta,cr):
        tau1,tau2,beta1,betau = self.adap_conf
        r1,r2,r3,r4 = np.random.uniform(0,1,4)
        if(r2 < tau1): beta = round(beta1 + r1 * betau,3) # else, keep the beta same
        if(r4 < tau2): cr = r3
        return beta,cr

    def mutation_rand_1_z(self,x0,x1,x2,beta):
        z = x0+beta*(x1-x2)
        z = np.where(z<0,0,z)
        #z = np.where(z>1,1,z)
        return z

    def crossoverRandomSwap(self,parent,u,cr):
        # the first one is with min len
        swap = np.random.uniform(0,1,self.R.shape[1])
        swaplayer = np.random.choice(range(self.R.shape[1]),size=1)[0]
        for i,s in enumerate(swap):
            if(s<=cr or i==swaplayer):
                parent[i] = u[i]
        return parent

    def crossoverUnif(self,parent,u,cr):
        # the first one is with min len
        swap = np.random.uniform(0,1,self.R.shape[1])[0]
        swaplayer = np.random.choice(range(self.R.shape[1]),size=1)[0]
        for i in range(len(parent)):
            if(swap<=cr or i==swaplayer):
                parent[i] = u[i]
        return parent


    def run(self,beta=0.5,cr=0.9):
        totalStart=time.time()
        regenCount=0
        stop=False
        current_gen=self.population
        scores = np.zeros((self.pplSize))
        #initial Run
        print('JDE Initial Run Start')
        for i in range(len(current_gen)):
            s = self.fit(current_gen[i],self.R)
            scores[i]=s
        print('JDE Initial Run End')
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        bestGen = 0
        print(f'JDE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}')
        #Generation Run
        i=0
        while (i < self.maxiter):
            structureStatistic=np.zeros((self.pplSize,4))
            updatecount=0
            start=time.time()
            print(f'Gen {i} Run Start')
            betas = np.ones(self.pplSize)*beta
            crs = np.ones(self.pplSize)*cr
            for j in range(self.pplSize):
                parent = current_gen[j]
                # factors
                betas[j],crs[j] = self.jde_params(betas[j],crs[j])
                # mutation
                idx0,idx1,idxt = np.random.choice(range(0,self.pplSize),3,replace=False)
                unitvector = self.mutation_rand_1_z(current_gen[idxt],current_gen[0],current_gen[1],betas[j])
                # crossover
                nextGen = self.crossoverUnif(parent,unitvector,crs[j])
                #print(f'Next Gen: {nextGen}')
                structureStatistic[j,0]= np.mean(nextGen)
                structureStatistic[j,1]= np.median(nextGen)
                structureStatistic[j,2]= np.quantile(nextGen,0.25)
                structureStatistic[j,3]= np.quantile(nextGen,0.75)
                s= self.fit(nextGen,j)
                if(s<scores[j]):
                    updatecount+=1
                    scores[j]=s
                    current_gen[j]=nextGen
                if(updatecount==0):
                    print(f'JDE Run {i:3d} - ReGen Population')
                    regenCount+=1
                    top10=np.argsort(scores)[:10]
                    top10=current_gen[top10]
                    newPopulation=np.random.uniform(0,1,size=(self.pplSize-10,self.R.shape[1]))
                    current_gen=np.vstack((top10,newPopulation))
            print(f'JDE Gen {i} Run End')
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
            elif(currentbest==overallBest):
                stop=True
            print(f'JDE Run {i:3d} CurrentBest: {currentbest:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, Generation RunTime: {(end-start):10.8f}, ReGenCount: {regenCount:3d}')
            print(f'a Statistic| genMeanMedian : {genMeanNode:5.4f} | genMedianMedian  :{genMedianNode:5.4f} | gen25q : {genq25Node:5.4f} | gen27q : {genq75Node:5.4f} | updatecount : {updatecount:3d}' )
            print(f's Statistic| Mean : {currentmean:5.4f} | Median  :{currentmedian:5.4f} | gen25q : {currentq25:5.4f} | gen27q : {currentq75:5.4f} | updatecount : {updatecount:3d}' )
            i+=1
        totalEnd=time.time()-totalStart
        print(f'JDE Run Completed : Best Score: {overallBest} , Config: {overallBestConfig}, find in Gen: {bestGen}, ReGenCount: {regenCount:3d}, totalRuntime: {(totalEnd):4.5f}')
        return
R=[[0.56,0.94,0.33,0.61,0.98,1,0.67,0.83,0.11,0.17,0.72,0.22,0.23,0.28,0.44,0.31,0.78,0.89,0.5,0.39],
[0.13,0.56,0.22,0.89,0.33,0.94,0.78,0.39,0.16,0.26,0.28,0.11,0.17,0.18,1,0.3,0.44,0.67,0.42,0.2],
[0.22,0.24,0.44,0.38,0.67,0.94,0.56,0.48,0.27,0.28,0.29,0.17,0.11,0.3,0.72,0.32,0.41,1,0.61,0.5],
[0.44,0.67,0.11,0.22,0.16,0.31,0.24,0.56,0.27,0.89,1,0.2,0.29,0.33,0.72,0.31,0.37,0.78,0.41,0.32],
[0.22,0.23,0.56,0.26,0.44,0.28,0.17,0.67,0.33,0.89,0.11,0.78,0.7,1,0.3,0.61,0.76,0.31,0.39,0.32]]
R=np.array(R)
b=np.array([0.1853,0.8533,0.8016,0.8696,0.4412])
jde = JDEV2_FRE(maxiter=800,R=R,b=b)
jde.run()

