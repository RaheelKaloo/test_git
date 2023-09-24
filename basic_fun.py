#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:34:59 2020

@author: Raheel
"""
import itertools
import numpy as np
import random
import pandas as pd
#import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score
from random import randint
from kmodes.kmodes import KModes
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.cluster import KMeans
from sklearn.metrics import cohen_kappa_score as Kapp
import constant as c

def feature_MI(coded_data,pred_lable,neighbor):
     feature_scores=mutual_info_classif(coded_data,pred_lable,
                                n_neighbors=neighbor, copy=True)
     attribute= coded_data.columns
     Mutual_information_dic=dict(zip(attribute, feature_scores))
     sorted_weight_MI=sorted(Mutual_information_dic,key=(lambda key:Mutual_information_dic[key]),reverse=True)     
     MI= sorted_weight_MI[:9]
     print("")
     return MI

def find_feature_MI( data,pred_lable):
    '''
    This function takes original data and the clustering result(labe)

    Returns :subset of features
    -------

    '''
    coded_data=encodeCategorical(data)
    FS=feature_MI(coded_data,pred_lable,10)
    return FS
        
    

def feature_fClassify(data,pred_lable):
     coded_data=encodeCategorical(data)
     _, feature_scores = f_classif(coded_data,pred_lable)
     attribute= coded_data.columns
     fClassify_dic=dict(zip(attribute, feature_scores))
     sorted_weight_F=sorted(fClassify_dic,key=(lambda key:fClassify_dic[key]),reverse=True)
     return  sorted_weight_F[:4]
     
     
# =============================================================================
# def select_feature(coded_data,pred_lable, neighbor, top_k):
#     _, result = f_classif(coded_data,pred_lable)
#     feature=mutual_info_classif(coded_data,pred_lable,
#                                 n_neighbors=neighbor, copy=True)
#     X_new = SelectKBest(chi2, k=top_k).fit_transform(coded_data, pred_lable)
# 
# =============================================================================
def encodeCategorical(tempdf):
    """
    This function convert categorial data to numerical 

    Parameters
    ----------
    tempdf : dataframe

    Returns
    -------
    Encod_df : 

    """
    for item in tempdf.columns:
        tempdf[item] = tempdf[item].astype('category')
    for item in tempdf.columns:
        tempdf[item] = tempdf[item].cat.codes
    return tempdf 

def encodeCategorical_FRRCO():
    tempdf = c. DATA
    print(len(tempdf.columns))
    
    for item in tempdf.columns:
        tempdf[item] = tempdf[item].astype('category')
    for item in tempdf.columns:
        tempdf[item] = tempdf[item].cat.codes
    
    return tempdf 

def create_combantion (size):
    """
    This function take int as size of dataframe 
    """
    names= list(range(0, size))
    all_pairs = list(itertools.combinations(names, 2))
    return all_pairs

def create_co_matrix(size, clusters, M):
    """
    Parameters
    ----------
    size : int,size of dataframe.
    clusters : a list, list of list each list is a cluster in partition.
    Returns
    -------
    adjacency_matrix : matrix of size n*n 
    """
    all_pair=create_combantion(size)
    adjacency_matrix = np.zeros(shape=(size,size))
    np.fill_diagonal(adjacency_matrix, 1)
    for pair in all_pair:
        p1= pair[0]
        p2=pair[1]
        pair_ocurance= 0.0
        for cluster in clusters:
            if (p1 in cluster )and (p2 in cluster):
                pair_ocurance +=1
        distance= float (pair_ocurance / M)
        adjacency_matrix[p1][p2] = 1.0 - distance
        adjacency_matrix[p2][p1] = 1.0- distance
        pair_ocurance =0.0
    return adjacency_matrix

def decode_data(dfVPRST):
    for col in dfVPRST:
        lsit_unique=dfVPRST[col].unique()
        for uq in lsit_unique:
            dfVPRST[col].replace(uq,(str(col)+str(uq)),inplace=True)
    dfVPRST.to_csv("hays_RST.csv",index=False) 
    #return dfVPRST

def create_co_matrix_weight(size, clusters,df_clusters_weight, M):
    """
    Parameters
    ----------
    size : int,size of dataframe.
    clusters : a list, list of list each list is a cluster in partition.
    Returns
    -------
    adjacency_matrix : matrix of size n*n 
    """
    all_pair=create_combantion(size)
    adjacency_matrix = np.zeros(shape=(size,size))
    np.fill_diagonal(adjacency_matrix, 1)
    for pair in all_pair:
        p1= pair[0]
        p2=pair[1]
        pair_ocurance= 0.0
        for cluster in clusters:
            if (p1 in cluster )and (p2 in cluster):
                pair_ocurance +=1
        distance= float (pair_ocurance / M)
        adjacency_matrix[p1][p2] = distance
        adjacency_matrix[p2][p1] = distance
        pair_ocurance =0.0
    return adjacency_matrix

def compute_NMI(perdict):
        true= c.True_class
        accurcyNMI_cl=normalized_mutual_info_score(true,perdict)
        print("accurcyNMI_cl---------",accurcyNMI_cl)



def do_subIntegration(matrix, num_clu,true,not_cluster):
    filter_true=[]
    filter_model=[]
    model = AgglomerativeClustering(affinity='precomputed', n_clusters=num_clu,
                                    linkage='complete').fit(matrix)
    
    for x in range(len(true)):
        if x not in not_cluster:
            filter_true.append(true[x])
            filter_model.append(model.labels_[x])
            
    
    accurcyNMI=normalized_mutual_info_score(filter_true,filter_model)
    print("Complete model",accurcyNMI)
    

def do_integration (matrix, num_clu,true):
    model = AgglomerativeClustering( n_clusters=num_clu,
                                    linkage='complete').fit(matrix)
   
    accurcyNMI_cl=normalized_mutual_info_score(true,model.labels_)
    print("Complete model",accurcyNMI_cl)
    accurcyNMI_cl = float("{:.4f}".format(accurcyNMI_cl))
    
    #-------------------------------------------------------------
    model_s = AgglomerativeClustering( n_clusters=num_clu,
                                    linkage='single').fit(matrix)
    
    accurcyNMI_sl=normalized_mutual_info_score(true,model_s.labels_)
    print("single model",accurcyNMI_sl)
    accurcyNMI_sl=float("{:.4f}".format(accurcyNMI_sl))
    
    #-------------------------------------------------------------
    model_a = AgglomerativeClustering( n_clusters=num_clu,
                                    linkage='average').fit(matrix)
    
    accurcyNMI_av=normalized_mutual_info_score(true,model_a.labels_)
    print("average model",accurcyNMI_av)
    accurcyNMI_av=float("{:.4f}".format(accurcyNMI_av))
    
    #-------------------------------------------------------------
    return accurcyNMI_cl ,accurcyNMI_sl,accurcyNMI_av,model.labels_
    
def spectralClustering(sim_matrix,CAS_K):
    """
    This method is used t
    Parameters
    ----------
    sim_matrix : .
    CAS_K : TYPE
        DESCRIPTION.

    Returns
    -------
    label : TYPE
        DESCRIPTION.
    cas_lists : TYPE
        DESCRIPTION.

    """
    X = np.array(sim_matrix)
    clustering = SpectralClustering(n_clusters= CAS_K,
    assign_labels="discretize",random_state=0).fit(X)
    label= list(clustering.labels_)
    cas_lists= [[] for i in range( CAS_K)]
    for x in range(len(label)):
        cas_lists[label[x]].append(x)
        
    return label,cas_lists    

def computeGK(Partition_E,data_size):

    len_item_square=0
    for item in Partition_E:
        len_item_square += pow(len(item) ,2)
    Gk_E= float(len_item_square / pow(data_size,2))
    return Gk_E

def cluster_subspace(ensemble_lib_size):
    base_clusters=[]
    for i in range (ensemble_lib_size):
        d=c.DATA
        rand_clusters=randint(4,10)
        d=d.sample(n=rand_clusters,axis=1)
        atrri=list(d.columns.values)
        d["id"]= d.index
        run_clusters= d.groupby(atrri)["id"].apply(list).tolist()
        base_clusters.append(run_clusters)
        return base_clusters
    
def do_clustering(data, num_clu, iter_max):
        """
    Parameters
    ----------
    data : dataFrame
    num_clu : int, number of cluster
    iter_max : int
    Returns
    inti=Huang
    init= Cao
    -------
    clusters : list, cluster_lable.
    """ 
        clusters=[]
        rand_clusters=randint(2,4)
        #randomm_runs=randint(2,3)
        print("rand clusters",rand_clusters)
        
        km = KModes(n_clusters= num_clu , init='random', n_init=iter_max, verbose=0)
        #km = KModes(n_clusters= 4, init='random', n_init=randomm_runs, verbose=0)
        km.fit_predict(data)
        centriod=list(km.cluster_centroids_)
        #print("clusteroid",km.cluster_centroids_)
        clusters=list(km.labels_)
        #print(set(km.labels_))
        return clusters,centriod
    
def doClusteringKmeans(newDF,number_cluster):
    #newDF.fillna(newDF.mean())
    rand_clusters=randint(2, 6)
    kmeans = KMeans(n_clusters= rand_clusters, random_state=0).fit(newDF)
    clusters=list(kmeans.labels_)  
    #cerntoid= list(kmeans.cluster_centroids_)  
    return clusters

def intersection(ind_cond, ind_deci): 
    '''
    This function returns a set of integration list

    Parameters
    ----------
    ind_cond : TYPE
        DESCRIPTION.
    ind_deci : TYPE
        DESCRIPTION.

    Returns
    -------
    lst3 : TYPE
        DESCRIPTION.

    '''
    temp = set(ind_deci) 
    lst3 = [value for value in ind_cond if value in temp] 
    return lst3 

def intersection_jaccard(x, y): 
    '''
    This function returns a set of integration list

    Parameters
    ----------
    ind_cond : TYPE
        DESCRIPTION.
    ind_deci : TYPE
        DESCRIPTION.

    Returns
    -------
    lst3 : TYPE
        DESCRIPTION.

    '''
    
    union=[]
    temp = set(x) 
    lst3 = [value for value in y if value in temp] 
    for i in x :
        union.append(i)
    for i in y :
        union.append(i)
    
    all_item= list(set(union))
    return float (len(lst3 ) / len(all_item))