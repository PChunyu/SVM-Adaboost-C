#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, train_test_split,StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import numpy as np
import pandas as pd
from svmpp4 import SVM4


def createSigleBoostingTree(x_train,y_train,sigma, D,C):
    svmcp = SVM4(x_train,y_train, sigma, C,0.001)
    svmcp.train()  
    sigleBoostTree = {}
    sigleBoostTree['model'] = svmcp
    pre=list(map(lambda i:int(svmcp.predict(i)),x_train))   
    e=sum([D[i] for i in range(len(x_train)) if (pre[i] != y_train[i])])
    sigleBoostTree['e'] = e
    sigleBoostTree['Gx'] = np.array(pre)    
    return sigleBoostTree
    
    
def createBosstingTree(x_train,y_train,treeNum,inic,sigma):
    m,mm = np.shape(x_train)
    finallpredict = [0] * m
    D = [1 / m] * m
    C=[inic]*m
    tree = []
    n=0
    for j in range(treeNum):
        curTree = createSigleBoostingTree(x_train,y_train,sigma, D,C)
        alphaa = 1/2 * np.log((1 -curTree['e']) / max(curTree['e'],1e-06) ) 
        Gx = curTree['Gx']
        D = np.multiply(D, np.exp(-1 * alphaa * np.multiply(y_train, Gx))) / sum(D)        
        suma=sum([D[i] for i in range(m) if (y_train[i] == 1 and Gx[i] != 1 )])
        sumb=sum([D[i] for i in range(m) if (y_train[i] == -1 and Gx[i] != -1 )])
        if (sumb > suma) : 
            D=[D[i]*sumb/suma if (y_train[i] == 1 and Gx[i] != 1 ) else D[i] for i in range(len(D))]
        D=D/sum(D)
        C=[C[k]*(1+m*D[k]) if Gx[k] != y_train[k] else C[k] for k in range(m)]
        curTree['alphaa'] = alphaa
        n=n+1
        tree.append(curTree)
        finallpredict += alphaa * Gx
        precision=metrics.precision_score(y_train,np.sign(finallpredict))
        recall=metrics.recall_score(y_train,np.sign(finallpredict))
        p=(recall*precision)**0.5
        f=metrics.f1_score(y_train,np.sign(finallpredict))
        acc=metrics.accuracy_score(y_train,np.sign(finallpredict))
        if recall == 1:    return tree
        print('n:%d,accuracy:%.4f,precision:%.4f, recall:%.4f,pmean:%.4f,fmean:%.4f'%(n,acc,precision,recall,p,f))
    return tree


def calc_all_need(x_test,y_test, tree):
    prob = []
    res={}
    for i in range(len(x_test)):
        result = 0
        for curTree in tree:
            alphaa = curTree['alphaa']
            svmcp = curTree['model']
            svm_pred  = int(svmcp.predict(x_test[i]))
            result += alphaa * svm_pred         
        prob.append(result)
        
    cm=metrics.confusion_matrix(y_test,np.sign(prob)) 
    specificity=cm[0][0]/(cm[0][1]+cm[0][0])
    precision=metrics.precision_score(y_test,np.sign(prob))
    recall=metrics.recall_score(y_test,np.sign(prob))
    p=(recall*precision)**0.5
    g=(recall*specificity)**0.5   
    f=metrics.f1_score(y_test,np.sign(prob))
    acc=metrics.accuracy_score(y_test,np.sign(prob))
    roc_auc = metrics.roc_auc_score(y_test, prob)
    
    res['G-Mean']=g
    res['F-Mean']=f
    res['P-Mean']=p
    res['Accuracy']=acc
    res['AUC']=roc_auc 
    res['Recall']=recall
    return   res 

if __name__ == '__main__':   
    start = time.time()
    name='c_after'
    s='D:/practise/TF/Pageblock/pa50.csv'
    df = pd.read_csv(s,header=None)
    x=df.iloc[:,1:]
    y=df[0]
    dataSet=np.array(x)
    labels=np.array(y)
    treeNum=25;inic=1;sigma=1
    KF=StratifiedKFold(n_splits=5,shuffle = True,random_state =321) 
    scale= StandardScaler()
    i = 0   
    calc={}
    for train,test in KF.split(dataSet,labels):
        x_train,x_test=np.array(dataSet)[train],np.array(dataSet)[test]
        y_train,y_test=np.array(labels)[train],np.array(labels)[test]  
        scale.fit(x_train)
        x_train_scaled=scale.transform(x_train)
        x_test_scaled=scale.transform(x_test) 
        tree=createBosstingTree(x_train_scaled,y_train,treeNum,inic,sigma)
        res=calc_all_need(x_test_scaled,y_test, tree)
        calc[i]={}
        for key in res.keys():
            calc[i][key]=res[key]
        i += 1 
        print(res)
    pp = []
    for key in calc[0]:
        pp.append(key)
    for i in pp:
        a = []
        for key0 in calc:
            for key1 in calc[key0]:
                if i == key1:
                    a.append(calc[key0][i])
        a=pd.DataFrame(a)
        print(i)
        print('mean:%.4f'%(a[0].mean()))
  
    print(name,s.split("/")[-2],treeNum,sigma)    
    end = time.time()
    print('time span:', end - start)        

