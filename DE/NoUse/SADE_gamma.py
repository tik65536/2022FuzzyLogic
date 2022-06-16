#!/bin/python3
import numpy as np
#from torchsummary import summary
import time

def minmax(x,y):
    z = np.max(np.minimum(x,y),axis=1)
    return z.reshape(1,-1)

class SADE_FRE():
    def __init__(self, initSize=200,maxiter=100,error=1e-03,R=None,b=None):
        self.best=[]
        self.mean=[]
        self.error = error
        self.pplSize = initSize
        self.maxiter = maxiter
        self.R=R
        self.b=b.reshape(1,-1)
        self.population = np.random.gamma(0.5,1,(self.pplSize,self.R.shape[1]))

    def fit(self,config,id_):
        config = config.reshape(1,-1)
        config = np.repeat(config,self.R.shape[0],axis=0)
        c = minmax(config,self.R)
        return np.sum((self.b-c)**2)

    def mutation_rand_1_z(self,x0,x1,x2,beta):
        z = x0+beta*(x1-x2)
        z = np.where(z<0,0,z)
        z = np.where(z>1,1,z)
        return z

    def crossoverUnif(self,parent,u,cr):
        # the first one is with min len
        swap = np.random.uniform(0,1,self.R.shape[1])[0]
        swaplayer = np.random.choice(range(self.R.shape[1]),size=1)[0]
        for i in range(len(parent)):
            if(swap<=cr or i==swaplayer):
                parent[i] = u[i]
        return parent

    def run(self):
        totalStart=time.time()
        regenCount=0
        upBound=np.full(self.R.shape[0],-1)
        lowBound=np.full(self.R.shape[0],-1)
        stop=False
        current_gen=self.population
        scores = np.zeros((self.pplSize))
        #initial Run
        print('SADE Initial Run Start')
        for i in range(len(current_gen)):
            s = self.fit(current_gen[i],i)
            scores[i]=s
        print('SADE Initial Run End')
        currentbest = np.min(scores)
        overallBest = currentbest
        currentmean = np.mean(scores)
        currentbestidx = np.argmin(scores)
        overallBestConfig = current_gen[currentbestidx]
        upBound = (1-overallBestConfig)/2+overallBest
        lowBound = (overallBestConfig-0)/2+overallBest
        bestGen = 0
        print(f'SADE Init Run Best: {currentbest}, Mean: {currentmean}, ID:{currentbestidx}')
        # initial factors
        p1,beta,crm = 0.5,0,0.5
        crs = []
        progress = []
        #Generation Run
        i=0
        while (i<self.maxiter):
            structureStatistic=np.zeros((self.pplSize,4))
            updatecount=0
            start=time.time()
            ns1,nf1,ns2,nf2 = 0,0,0,0
            print(f'SADE Gen {i} Run Start')
            for j in range(self.pplSize):
                parent = current_gen[j]
                # factors
                while beta <= 0 or beta > 2: beta = np.random.normal(loc=0.5,scale=0.3,size=1)[0]
                if i%5==0: cr = np.random.normal(loc=crm,scale=1,size=1)[0]
                r = np.random.uniform(low=0,high=1,size=1)[0]
                # strategy selection + mutation
                idx0=-1
                idx1=-1
                idxt=-1
                if r <= p1:
                    idx0,idx1,idxt = np.random.choice(range(0,self.pplSize),3,replace=False)
                else:
                    idxt = np.argmax(scores)
                    idx0,idx1 = np.random.choice(np.delete(np.arange(self.pplSize),idxt),2,replace=False)
                unitvector = self.mutation_rand_1_z(current_gen[idxt],current_gen[0],current_gen[1],beta)
                # crossover
                child = self.crossoverUnif(parent,unitvector,cr)
                structureStatistic[j,0]= np.mean(child)
                structureStatistic[j,1]= np.median(child)
                structureStatistic[j,2]= np.quantile(child,0.25)
                structureStatistic[j,3]= np.quantile(child,0.75)
                # selection
                s= self.fit(child,j)
                if(s < scores[j]):
                    updatecount+=1
                    scores[j]=s
                    current_gen[j]=child
                    crs.append(cr)
                    if r <= p1: ns1 += 1
                    else: ns2 += 1
                else:
                    if r <= p1: nf1 += 1
                    else: nf2 += 1
            if i%5==4:
                crm = np.mean(crs)
                crs = []
            if(ns2*(ns1+nf1)+ns1*(ns2+nf2) == 0): p1 = 0
            else: p1 = (ns1*(ns2+nf2))/(ns2*(ns1+nf1)+ns1*(ns2+nf2))
            print(f'SADE Gen {i} Run End')
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
                print(f'SADE Run {i:3d} - ReGen Population')
                regenCount+=1
                top = np.argmin(scores)
                top = current_gen[top]
                combine=np.hstack((upBound,lowBound))
                k = np.mean(combine)**2/np.var(combine)
                theta = np.var(combine)/np.mean(combine)
                current_gen = np.random.gamma(k,theta,(self.pplSize,self.R.shape[1]))
                #current_gen = np.vstack((top,current_gen))
                current_gen = np.where(current_gen>1,1,current_gen)
                #current_gen = np.where(current_gen<0,0,current_gen)
                for k in range(self.pplSize):
                    scores[k] = self.fit(current_gen[k],k)
            print(f'SADE_gamma Run {i:3d} CurrentBest: {currentbest:10.8f}, OverallBest: {overallBest:10.8f}/{bestGen:3d}, Generation RunTime: {(end-start):10.8f}')
            print(f'a Statistic| gen25q: {genq25Node:5.4f} | genMeanMedian : {genMeanNode:5.4f} | genMedianMedian  :{genMedianNode:5.4f} | gen27q : {genq75Node:5.4f} | updatecount : {updatecount:3d}' )
            print(f's Statistic| gen25q: {currentq25:5.4f} | Mean : {currentmean:5.4f} | Median  :{currentmedian:5.4f} | gen27q : {currentq75:5.4f} | updatecount : {updatecount:3d} | ReGenCount: {regenCount:3d}' )
            i+=1
        totalEnd=time.time()-totalStart
        print(f'SADE Run Completed : Best Score: {overallBest} ,  find in Gen: {bestGen}, ReGenCount: {regenCount:3d}, totalRuntime: {(totalEnd):4.5f}')
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
sade = SADE_FRE(maxiter=10000,R=R,b=b,error=0.0001)
sade.run()
