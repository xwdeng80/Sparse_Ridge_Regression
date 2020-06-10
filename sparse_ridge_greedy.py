#!/usr/bin/python

# Copyright 2016, Gurobi Optimization, Inc.

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import sys
import math
import random
import itertools
from gurobipy import *
import os
import xlrd
import csv
import random
import numpy as np
import scipy as sp
import pandas as pd
import time



global df
df = pd.DataFrame(columns=('Case #','n','p','k','lambda','Greedy error', 'Greedy time', 'Greedy Beta','Greedy Support', '# of correct prediction'))

def compute_beta_val(zchoose):
    mm= Model()#Model (11)
    # Create variables  
    mbetavars = mm.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='mbeta')
    mmuvars = mm.addVars(p, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='mmu')
    #add objective
    obj = LinExpr()
    obj = 0;
    for i in range(n):
        obj +=1.0/float(n)*(y[i]-quicksum(X[i][j]*mbetavars[j] for j in range(p)))*(y[i]-quicksum(X[i][j]*mbetavars[j] for j in range(p)))
    for j in range(p):
        obj +=lamval* mmuvars[j]

    mm.setObjective(obj, GRB.MINIMIZE)
    # Adding constraints
    for j in range(p):
        mm.addConstr(mbetavars[j]*mbetavars[j],GRB.LESS_EQUAL,zchoose[j]*mmuvars[j],'mbeta z '+str(j))
    mm.update()
    # Optimize model
    #m.Params.lazyConstraints = 0
    mm.Params.OutputFlag=0
    mm.optimize()
    mbetavals = mm.getAttr('x', mbetavars)
    #zvals = m.getAttr('x', zvars)
    return mbetavals, mm.objVal


def compute_beta_val2(SS):
#    X1=[]
#    for i in SS:
 #       X1.append(X[i][:])
 #   print len(X1)
    Xnew=np.matrix(np.array(X))
    Xnew=Xnew[:,SS]
 #   Xnew=Xnew.T
    Ynew=np.matrix(np.array(y))
    Ynew=Ynew.T

    kk=len(SS)
    XXnew=Xnew.T
    mv=1.0/float(n)*XXnew*XXnew.T+lamval*np.matrix(np.identity(kk))
    mbetaval=1.0/float(n)*np.linalg.inv(mv)*XXnew*Ynew
 #   mbetavals=[0.0]*p
 #   count1=0
 #   for i in range(kk): 
 #       mbetavals[i]=float(mbetaval[count1])
 #       count1=count1+1
    
    #zvals = m.getAttr('x', zvars)
    return mbetaval

def compute_val(SS):
    Xnew=np.matrix(np.array(X))
    Xnew=Xnew.T
    Ynew=np.matrix(np.array(y))
    Ynew=Ynew.T
    S = [[(Xnew[i].T * Xnew[i]) for i in range(p)]]
    S=S[0]
    value=sum(S[j] for j in SS)+lamval*float(n)*np.matrix(np.identity(n))
    value=float((lamval*Ynew.T*np.linalg.inv(value)*Ynew))

    return value

def update_dif(Ainv,Ynew, xnew):
    #print math.pow(Ynew.T*Ainv* xnew,2.0)
    ab=Ainv* xnew
    value=-(math.pow(Ynew.T*ab,2.0))/(1.0+xnew.T*ab)
    value=float(value)

    return value    


k=10 ## number of sparsity

Strue=[]
i = 1
reader = csv.reader(open("beta_info_p1000k10.csv"))
for line in reader:  # iterate over the lines in the csv
    rows=[]
    if i>=2:
        Strue.append(float(line[0]))
    i=i+1



y = []
X =[]
i = 1
reader = csv.reader(open("data_n500p1000k10_rep1.csv"))
for line in reader:  # iterate over the lines in the csv
    rows=[]
    if i>=2:
        for j in range(len(line)):  # check if the 2nd element is one you're looking for
            if j==0:
                y.append(float(line[0]))
            else:
                rows.append(float(line[j]))
        X.append(rows)
    i=i+1


random.seed(1)
n=len(y)
p=len(X[0][:])
#       tuning parameter
lamval=0.08

print('n= %g, p=%g, k=%g' %(n,p,k))

####################################Greedy
print('##########################Greedy Results')
start = time.time()

zchoose=[0.0]*p
S=[]
T=range(p)

Xnewval=np.matrix(np.array(X))
Xnewval=Xnewval.T
Ynewval=np.matrix(np.array(y))
Ynewval=Ynewval.T
greedyerror=lamval*Ynewval.T*Ynewval/(lamval*float(n))

Ainvx=np.zeros((p, n))
yAinvx=np.zeros((p, 1))
xAinvx=np.zeros((p, 1))
for j in range(p):
    xnew=Xnewval[j]
    Ainvx[j]=xnew/(lamval*float(n))
    yAinvx[j]=float(xnew/(lamval*float(n))* Ynewval)
    xAinvx[j]=float(xnew/(lamval*float(n))* xnew.T)



while len(S)<k:
    bestj=0
    valnew=1e20
    
 
    for j in range(len(T)):
        ind =T[j]
        value=-(math.pow(yAinvx[ind],2.0))/(1.0+xAinvx[ind])
        val=float(value)
        if valnew>val:
            valnew=val
            
            bestj=j
            
    greedyerror=greedyerror+lamval*valnew
    S.append(T[bestj])
    ind=T[bestj]
    del T[bestj]
    bestxnew=Xnewval[ind]
    Ainvxs=np.copy(Ainvx[ind])
    xAinvxs=np.copy(xAinvx[ind])
    for j in range(p):
        Ainvx[j]=Ainvx[j]-float(Ainvx[j]*bestxnew.T/(1.0+xAinvxs))*Ainvxs
        yAinvx[j]=float(Ainvx[j]* Ynewval)
        xAinvx[j]=float(Ainvx[j]* Xnewval[j].T)
greedyerror=float(greedyerror)
print('greedy is done')

for j in S:
    zchoose[j]=1.0
greedybeta=compute_beta_val2(S)

bestgreedyS=np.array(S)+1

greedytime = time.time() - start
print('')

print('greedy time: %g' % greedytime)

count=0
for i in bestgreedyS:
    if i in Strue:
        count=count+1
print('number of right positions found: %g' % count)
print('')




