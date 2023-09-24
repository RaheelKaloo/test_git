# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:42:07 2023

@author: DELL
"""
import warnings
from sklearn import metrics
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from itertools import permutations
from sklearn.metrics import confusion_matrix
import constant as c
from random import randint
import random
from sklearn.cluster import AgglomerativeClustering
import ac_class as ac
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import entropy
from math import log, e
from sklearn.metrics import jaccard_score
numeric_data="derma_RST1.csv"
class_data= "derma_class.csv"
cateData="derma_RST.csv"
llso=[]
allequavlance_cond=[]
allNonequavlance_cond=[]
emsemble_mode_list={}
unique_features_ensemble={}


def create_coMatix(list_clusters):
    #list_cluster= sum(list_list_clusters)
    df= pd.read_csv("derma_RST.csv")
    co_matrix = np.zeros(shape=(c.DATA_SIZE,c.DATA_SIZE))
    for cluster in list_clusters:
        perm = permutations(cluster, 2) 
        for pair in perm:
            if pair[0] in range(c.DATA_SIZE) and  pair[1] in range(c.DATA_SIZE):
                oi= pair[0]
                oj= pair[1]
                xi= df.iloc[oi]
                xj= df.iloc[oj]
                # print(pair[0])
                # print(pair[1])
                score=jaccard_score(xi, xj, average="weighted")
                
                co_matrix[oi][oj] +=  score
   # co_matrix= co_matrix / len(allequavlance_cond)
    return co_matrix
def find_cluster_features_entropy(temp_clusters,entropy_threshold):
   
   
    
    print("cluster size:", len(temp_clusters))
    clusters_reletvive_featutres=[]

    
    
    for col in temp_clusters.columns:
        value,counts = np.unique(temp_clusters[col], return_counts=True)
        col_entropy=entropy(counts, base=10)
        if col_entropy <=entropy_threshold :
            clusters_reletvive_featutres.append(col)
    print(clusters_reletvive_featutres)

    temp_clusters["id"]= temp_clusters.index
    if len(clusters_reletvive_featutres) >=5:
        equavlance_cond = (temp_clusters.groupby(clusters_reletvive_featutres)['id'].apply(list)).tolist()
        allNonequavlance_cond.append(equavlance_cond)
        for q in equavlance_cond:
            if len(q)>2:
                allequavlance_cond.append(q)




def main():

   
    df= pd.read_csv(numeric_data,header=None)
   
    km = KModes(n_clusters=c.Number_cluster, init="cao" ,verbose=0)

    clusters = km.fit_predict(df)
   
    df["class"]=clusters
    delat_range=[0.1,0.2,0.3]
    for delta in delat_range:
        print("-----------------------Delata---------------")
        for k in range(c.Number_cluster): # for each base clustering pi analysis 
            temp_df= df[df["class"]==k] # for each cluster in pi
            temp_df.drop(["class"], axis=1, inplace=True)
            find_cluster_features_entropy(temp_df,delta)
           
           
main()
co=create_coMatix(allequavlance_cond)
Agg_hc = AgglomerativeClustering(n_clusters =c.Number_cluster, affinity = 'l2', linkage = 'average')
print("---------------------------------------------")
clusters = list(Agg_hc.fit_predict(co)) # 
accurcyNMI_cl=normalized_mutual_info_score(c.True_class,clusters)
print("NMI,",accurcyNMI_cl)
ac.permutations_matrix(c.True_class,clusters)
