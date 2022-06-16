#!/bin/python3
import numpy as np
import time

def minmax(x,y):
    z = np.max(np.minimum(x,y),axis=1)
    return z.reshape(1,-1)

class JADE_FRE():
    def __init__(self, initSize=200,maxiter=100,error=1e-03,R=None,b=None):
        self.best=[]
        self.mean=[]
        self.error = error
        self.pplSize = initSize
        self.maxiter = maxiter
        self.R=R
        self.b=b.reshape(1,-1)
        self.population = np.random.uniform(0,1,(self.pplSize,self.R.shape[1]))
        self.adap_conf = (0.1,0.1,0.9)


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

    def find_pbest(self,scores,p):
        pi = int(p*len(scores)+1) # it is taken p% out of length and then ceiling
        fits = scores.copy() # keep scores and its indexes
        fits = sorted(fits)[::-1] # sort fits desc
        idx = np.random.choice(range(pi),1,replace=False)[0] # random index for sorted fits
        return list(scores).index(fits[idx]) # return the index from sorted

    def self_adaptive_beta(self,beta):
        tau,beta1,betau = self.adap_conf
        r1,r2 = np.random.uniform(0,1,2)
        if(r2 < tau): beta = round(beta1 + r1 * betau,3) # else, keep the beta same
        return beta

    def mutation_pbest_1_z(self,p,x0,x1,x2,beta):
        z = x0+beta*(x0-p)+beta*(x1-x2)
        z = np.where(z<0,0,z)
        z = np.where(z>1,1,z)
        return z

    def crossoverRandomSwap(self,parent,u):
        # the first one is with min len
        swap = np.random.choice(2,self.R.shape[1])
        for i,s in enumerate(swap):
            if(s!=0):
               parent[i]=u[i]
        return parent

    def run(self,beta=0.5,p=0.2):
        totalStart=time.time()
        stop=False
        regenCount=0
        upBound=np.full(self.R.shape[0],-1)
        lowBound=np.full(self.R.shape[0],-1)
        current_gen=self.population
        scores = np.zeros((self.pplSize))
        #initial Run
        print('JADE Initial Run Start')
        for i in range(len(self.population)):
            s = self.fit(current_gen[i],i)
            scores[i]=s
        print('JADE Initial Run End')
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        upBound = (1-overallBestConfig)/2+overallBest
        lowBound = (overallBestConfig-0)/2+overallBest
        bestGen = 0
        print(f'JADE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}')
        #Generation Run
        i=0
        while (i<self.maxiter):
            structureStatistic=np.zeros((self.pplSize,4))
            updatecount=0
            start=time.time()
            print(f'JADE Gen {i} Run Start')
            betas = np.ones(self.pplSize)*beta
            for j in range(self.pplSize):
                parent = current_gen[j]
                # factors
                betas[j] = self.self_adaptive_beta(betas[j])
                # mutation
                tidx = self.find_pbest(scores,p)
                idx1,idx2 = np.array(np.random.choice(np.delete(np.arange(self.pplSize),tidx),2,replace=False),dtype=int)
                unitvector = self.mutation_pbest_1_z(current_gen[j],current_gen[tidx],current_gen[idx1],current_gen[idx2],betas[j])
                # crossover
                nextGen = self.crossoverRandomSwap(current_gen[j],unitvector)
                structureStatistic[j,0]= np.mean(nextGen[1:-1])
                structureStatistic[j,1]= np.median(nextGen[1:-1])
                structureStatistic[j,2]= np.quantile(nextGen[1:-1],0.25)
                structureStatistic[j,3]= np.quantile(nextGen[1:-1],0.75)
                s = self.fit(nextGen,j)
                if(s<scores[j]):
                    updatecount+=1
                    scores[j]=s
                    current_gen[j]=nextGen

            print(f'JADE Gen {i} Run End')
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
            if((currentbest>overallBest) and (abs(overallBest-currentbest) < self.error)):
                currentbestConfig=current_gen[currentbestidx]
                idx=np.where(overallBestConfig<=currentbestConfig)[0]
                upBound[idx]=currentbestConfig[idx]
                lowBound[idx]=overallBestConfig[idx]
                idx=np.where(overallBestConfig>currentbestConfig)[0]
                upBound[idx]=overallBestConfig[idx]
                lowBound[idx]=currentbestConfig[idx]
            if(currentbest<overallBest):
                currentbestConfig=current_gen[currentbestidx]
                idx=np.where(overallBestConfig<=currentbestConfig)[0]
                upBound[idx]=currentbestConfig[idx]
                lowBound[idx]=overallBestConfig[idx]
                idx=np.where(overallBestConfig>currentbestConfig)[0]
                upBound[idx]=overallBestConfig[idx]
                lowBound[idx]=currentbestConfig[idx]
                overallBest=currentbest
                overallBestConfig = current_gen[currentbestidx]
                bestGen = i
            if(updatecount==0):
                top=np.argsort(scores)[:10]
                top = current_gen[top]
                mean = np.mean(top)
                current_gen = np.random.default_rng().normal(mean,0.2,(self.pplSize-10,self.R.shape[1]))
                current_gen = np.vstack((top,current_gen))
                current_gen = np.where(current_gen>1,1,current_gen)
                current_gen = np.where(current_gen<0,0,current_gen)
                regenCount+=1
                for k in range(self.pplSize):
                    scores[k] = self.fit(current_gen[k],k)
            print(f'JADE Run {i:3d} CurrentBest: {currentbest:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, Generation RunTime: {(end-start):10.8f}')
            print(f'a Statistic| gen25q: {genq25Node:5.4f} | genMeanMedian : {genMeanNode:5.4f} | genMedianMedian  :{genMedianNode:5.4f} | gen27q : {genq75Node:5.4f} | updatecount : {updatecount:3d}' )
            print(f's Statistic| gen25q: {currentq25:5.4f} | Mean : {currentmean:5.4f} | Median  :{currentmedian:5.4f} | gen27q : {currentq75:5.4f} | updatecount : {updatecount:3d} | ReGenCount: {regenCount:3d}' )
            i+=1
        totalEnd=time.time()-totalStart
        print(f'JADE Run Completed : Best Score: {overallBest} , find in Gen: {bestGen}, ReGenCount: {regenCount:3d}, totalRuntime: {(totalEnd):4.5f}')
        print(f'Best: {overallBestConfig}')
        print(f'Up  : {upBound}')
        print(f'Low : {lowBound}')
        return

R=[[0.56,0.94,0.33,0.61,0.98,1,0.67,0.83,0.11,0.17,0.72,0.22,0.23,0.28,0.44,0.31,0.78,0.89,0.5,0.39],
[0.13,0.56,0.22,0.89,0.33,0.94,0.78,0.39,0.16,0.26,0.28,0.11,0.17,0.18,1,0.3,0.44,0.67,0.42,0.2],
[0.22,0.24,0.44,0.38,0.67,0.94,0.56,0.48,0.27,0.28,0.29,0.17,0.11,0.3,0.72,0.32,0.41,1,0.61,0.5],
[0.44,0.67,0.11,0.22,0.16,0.31,0.24,0.56,0.27,0.89,1,0.2,0.29,0.33,0.72,0.31,0.37,0.78,0.41,0.32],
[0.22,0.23,0.56,0.26,0.44,0.28,0.17,0.67,0.33,0.89,0.11,0.78,0.7,1,0.3,0.61,0.76,0.31,0.39,0.32]]
R=np.array(R)
b=np.array([0.1853,0.8533,0.8016,0.8696,0.4412])
jade = JADE_FRE(maxiter=10000,R=R,b=b)
jade.run()
