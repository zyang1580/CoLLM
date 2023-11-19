"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import torch.utils.data


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    # @property
    # def m_users(self):
    #     raise NotImplementedError
    
    # @property
    # def n_items(self):
    #     raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class GnnDataset(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self,config, path="../data/gowalla"):
        # train or test
        # cprint(f'loading [{path}]')
        print("loading: ", path)
        self.split = config.A_split
        self.folds = config.A_n_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']


        train_file = path+"train_ood2.pkl"

        valid_file = path+"valid_ood2.pkl"
        test_file = path + "test_ood2.pkl"
        self.path = path
        
        self.traindataSize = 0
        self.testDataSize = 0


        self.train = pd.read_pickle(train_file)[['uid','iid','label']]
        self.train.columns = ['user','item','label']
        self.valid = pd.read_pickle(valid_file)[['uid','iid','label']]
        self.valid.columns = ['user','item','label']
        self.test = pd.read_pickle(test_file)[['uid','iid','label']]
        self.test.columns = ['user','item','label']

        # self.train = pd.read_csv(train_file)[['user','item','lables']]
        # self.valid = pd.read_csv(valid_file)[['user','item','lables']]
        # self.test = pd.read_csv(test_file)[['user','item','lables']]

        self.m_users = 1 + max([self.train['user'].max(),self.valid['user'].max(),self.test['user'].max()])
        self.n_items = 1 + max([self.train['item'].max(),self.valid['item'].max(),self.test['item'].max()] )
        
        self.testDataSize = self.test.shape[0]
        self.validDataSize = self.valid.shape[0]
        self.train_size = self.train.shape[0]


       
        
        self.Graph = None
        print(f"{self.train_size} interactions for normal training")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.m_users} users, {self.n_items} items")
        print(f"{config.dataset} Sparsity : {(self.validDataSize + self.testDataSize+self.train_size) / self.m_users / self.n_items}")

        # (users,items), bipartite graph
        # self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
        #                               shape=(self.m_users, self.n_items))
        # self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        # self.users_D[self.users_D == 0.] = 1
        # self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        # self.items_D[self.items_D == 0.] = 1.
        # # pre-calculate
        # self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # self.__testDict = self.__build_test()
        self._register_graph()
        
        print(":%s is ready to go"%(config.dataset))
    
    def _register_graph(self):
        self.getSparseGraph_mode_a2("graph")
        

 
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.m_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.m_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().cuda())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index,data,torch.Size(coo.shape))
        

        
    def getSparseGraph_mode_a2(self,mode):
        pos_train = self.train[self.train['label']>0].values.copy()
        pos_train[:,1] += self.m_users
        self.trainUser  = self.train['user'].values.squeeze()
        self.trainItem = self.train['item']
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_'+mode+'.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                pos_train_t = pos_train.copy()
                pos_train_t[:,0] = pos_train[:,1]
                pos_train_t[:,1] = pos_train[:,0]
                pos = np.concatenate([pos_train,pos_train_t],axis=0)

                adj_mat = sp.csr_matrix((pos[:,2], (pos[:,0],pos[:,1])), shape=(self.m_users+self.n_items, self.m_users+self.n_items))
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_'+mode+'.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                print("don't split the matrix")
        return self.Graph




    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    

    def generate_train_dataloader(self,batch_size=1024):
        '''
        generate minibatch data for full training and retrianing
        '''
        data = torch.from_numpy(self.train[['user','item','lables']].values)
        train_loader = torch.utils.data.DataLoader(data,shuffle=True,batch_size=batch_size,drop_last=False,num_workers=2)
        return train_loader