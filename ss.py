import numpy as np
import tensorflow as tf
import pandas as pd
import _pickle as pkl
import matplotlib.pyplot as plt
from amirata_functions import *
from UKB_CNN import UKB_CNN
import scipy.stats
import itertools

def impute_data(inputs,imputes,impute_method="simple"):
    if type(inputs)!= list:
        inputs = [inputs]
        imputes = [imputes]
    if len(imputes)!=len(inputs):
        raise ValueError("Different number of inputs and imputes!")
    returns = []
    if impute_method=="simple":
        for inp,imp in zip(inputs,imputes):
            inp[np.isnan(inp)] = -100
            returns.append(inp*(inp!=-100)+imp*(inp==-100))
    return returns

def one_hotize(data):
    lengths = []
    values=[]
    for counter in range(data.shape[-1]):
        values.append(np.unique(data[:,counter]))
        lengths.append(len(values[-1]))
    number = data.shape[0]
    cats_onehot=np.zeros((number,sum(lengths))).astype(int)
    cumsum = 0
    col_temp=np.zeros(number).astype(int)
    for counter in range(data.shape[-1]):
        for i,value in enumerate(values[counter]):
            col_temp[data[:,counter]==value]=i
        cats_onehot[np.arange(number),cumsum+col_temp]=1
        cumsum+=lengths[counter]
    return cats_onehot,lengths,values

with open("UKB.pkl","rb") as inputs:
    dic = pkl.load(inputs)
labels = dic["labels"].copy()
diseases=dic["diseases"]
cats = dic["cats"][:,np.where(np.sum(dic["cats"]!=-100,0)>450000)[0]]
floats = dic["floats"][:,np.where(np.sum(dic["floats"]!=-100,0)>450000)[0]]
cats_mode = [dic["cats_mode"][0][np.where(np.sum(dic["cats"]!=-100,0)>450000)[0]]]
floats_mean = [dic["floats_mean"][0][np.where(np.sum(dic["floats"]!=-100,0)>450000)[0]]]
scores = dic["scores"]
dic = {}
cats_imp, floats_imp = impute_data([cats,floats],[cats_mode,floats_mean])
floats=0
cats=0
cats_imp = cats_imp.astype(int)
numericals = np.concatenate([floats_imp,scores],-1)
scores=0
floats_imp=0
nums_norm=(numericals-np.mean(numericals,0,keepdims=True))/(np.std(numericals,0,keepdims=True)+1e-8)
cats_onehot,lengths,values = one_hotize(cats_imp)
cats_imp=0
floats_imp=0
kfold=3
lrs = [5e-4,1e-3,5e-3]
do = [0.5,0.8]
reg=[1e-4,1e-3,1e-2]
cw=[1,1.1,1.2]
nl = [1,2,3]
hids = [256,512,1024]
machine=1
search_dis = np.arange(20,40)
save_dic={"params":[[]]*len(search_dis),"f1":np.zeros(len(search_dis)),"acc":np.zeros(len(search_dis)),
          "size":np.zeros(len(search_dis))}
for shomarande,disease in enumerate(search_dis):
    best_val = 0
    label = (labels[:,disease]>0).astype(int)
    pos = np.where(label>0)[0]
    ratio=1
    neg = np.random.choice(np.where(label==0)[0],ratio*len(pos),replace=False)
    print(len(neg),len(pos))
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    fold_size = int(0.9*len(pos)/kfold)
    data_size=fold_size*kfold
    for it in itertools.product(lrs,do,reg,cw,nl,hids):
        f1=0
        acc=0
        val=0
        for fold in range(kfold):
            inds_train = np.concatenate([pos[:fold*fold_size],pos[(fold+1)*fold_size:data_size],
                                         neg[:fold*fold_size],neg[(fold+1)*fold_size:data_size]])
            inds_val = np.concatenate([pos[fold*fold_size:(fold+1)*fold_size],neg[fold*fold_size:(fold+1)*fold_size]])
            np.random.shuffle(inds_val)
            np.random.shuffle(inds_train)
            tf.reset_default_graph()
            obj = UKB_CNN(name="gooz", num_length=nums_norm.shape[-1], cat_length=cats_onehot.shape[-1], 
                          num_classes=2, path="./gooz",hidden_sizes=[it[5]]*it[4],task_weights=1,num_output_layers=0,
                          output_hidden_sizes=None,learning_rate = it[0], dropout = it[1],  class_weight=[1.,it[3]],
                          activation="relu", reg = it[2], batch_norm=True, loss = "weighted_CE", num_tasks=1,
                          activation_param = 0)
            sess=get_session()
            val+=obj.optimize(sess=sess,training_data=(cats_onehot[inds_train],nums_norm[inds_train],labels[inds_train,disease]), 
                             validation_data=(cats_onehot[inds_val],nums_norm[inds_val],labels[inds_val,disease]),save=True,
                              load=False , epochs=50,batch_size= 128,tradeoff=False, verbose=False, save_always=False,
                             return_as_fit=False, early_stopping=False, initialize=True, measure='accuracy')
            f1 += obj.best_f1
            acc += obj.best_acc
        if val/kfold>best_val:
            best_val = val/kfold
            best_f1 = f1/kfold
            best_acc = acc/kfold
            best_it=it
        save_dic["params"][shomarande]=best_it
        save_dic["f1"][shomarande]=best_f1
        save_dic["acc"][shomarande]=best_acc
        save_dic["size"][shomarande]=2*len(pos)
        print(best_it,best_val)
        with open("params_onetask{}.pkl".format(machine),"wb") as output:
            pkl.dump(save_dic,output)
        
            
            

