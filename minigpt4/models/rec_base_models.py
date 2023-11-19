import numpy as np
import torch
import torch.nn as nn
# import torch.functional as F
import torch.nn.functional as F
import os


class Personlized_Prompt(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.padding_index=0
        # self.half()
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    def user_encoder(self,users, all_users=None):
        return F.one_hot(users, num_classes = self.item_num+self.user_num).float()
    def item_encoder(self,items, all_items=None):
        return F.one_hot(items + self.user_num, num_classes = self.item_num+self.user_num).float()

class Soft_Prompt(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.padding_index=0
        # self.half()
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    def user_encoder(self,users, all_users=None):
        u_ = torch.zeros_like(users).to(users.device)
        return F.one_hot(u_, num_classes = 2).float()
    def item_encoder(self,items, all_items=None):
        i_ = torch.ones_like(items).to(items.device)
        return F.one_hot(i_, num_classes = 2).float()


class random_mf(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.user_num = config.user_num
        self.item_num = config.item_num
        self.padding_index=0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
        print("creat random MF model, user num:", config.user_num, "item num:", config.item_num)
        # self._init_weights()
        # self.half()
    def _init_weights(self):
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight.data)
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    def user_encoder(self,users,all_users=None):
        # print("user max:", users.max(), users.min())
        return self.user_embedding(users)
    def item_encoder(self,items,all_items=None):
        # print("items max:", items.max(), items.min())
        return self.item_embedding(items)

class MatrixFactorization(nn.Module):
    # here we does not consider the bais term 
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)
        print("creat MF model, user num:", config.user_num, "item num:", config.item_num)

    def user_encoder(self,users,all_users=None):
        # print("user max:", users.max(), users.min())
        return self.user_embedding(users)
    def item_encoder(self,items,all_items=None):
        # print("items max:", items.max(), items.min())
        return self.item_embedding(items)
    
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    
    def forward(self,users,items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        matching = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return matching

class MF_linear(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.MF_model = model
        self.llama_proj = nn.Linear(self.MF_model.user_embedding.weight.shape[1],4096)
    def forward(self,users,items):
        user_embedding = self.MF_model.user_embedding(users)
        item_embedding = self.MF_model.item_embedding(items)
        user_embedding_ = self.llama_proj(user_embedding)
        item_embedding_ = self.llama_proj(item_embedding)
        matching = torch.mul(user_embedding_, item_embedding_).sum(dim=-1)
        return matching
    


"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""

class LightGCN(nn.Module):
    def __init__(self, 
                 config):
        super(LightGCN, self).__init__()
        self.config = config
        self.padding_index = 0
        # self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.config.user_num
        self.num_items  = self.config.item_num
        self.latent_dim = self.config.embed_size #['latent_dim_rec']
        self.n_layers = self.config.gcn_layers #['lightGCN_n_layers']
        self.keep_prob = self.config.keep_prob #['keep_prob']
        self.A_split = self.config.A_split #['A_split']
        self.dropout_flag = self.config.dropout
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config.pretrain == 0:
            # nn.init.xavier_uniform_(self.embedding_user.weight, gain=nn.init.calculate_gain('sigmoid'))
            # nn.init.xavier_uniform_(self.embedding_item.weight, gain=nn.init.calculate_gain('sigmoid'))
            # print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=self.config.init_emb)
            nn.init.normal_(self.embedding_item.weight, std=self.config.init_emb)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # self.Graph = self.dataset.Graph
        print(f"lgn is already to go(dropout:{self.config.dropout})")
    
    def _set_graph(self,graph):
        self.Graph = graph.to(self.embedding_user.weight.device)
        self.Graph = self.Graph.to_sparse_csr() # necssary.... for half
        print("Graph Device:", self.Graph.device)

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        self.Graph = self.Graph.to(users_emb.device)
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def user_encoder(self, users, all_users=None):
        if all_users is None:
            all_users, all_items = self.computer()
        return all_users[users]
    
    def item_encoder(self, items, all_items=None):
        if all_items is None:
            all_users, all_items = self.computer()
        return all_items[items]
    


        
    def F_computer(self,users_emb,items_emb,adj_graph):
        """
        propagate methods for lightGCN
        """       
        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print("droping")
                raise NotImplementedError("dropout methods are not implemented")
                # g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = adj_graph        
        else:
            g_droped = adj_graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items



    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    

    def getEmbedding_v2(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        # neg_emb = all_items[neg_items]
        # users_emb_ego = self.embedding_user(users)
        # items_emb_ego = self.embedding_item(items)
        # neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, items_emb
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    def compute_bce_loss(self, users, items, labels):
        (users_emb, items_emb) = self.getEmbedding_v2(users.long(), items.long())
        matching = torch.mul(users_emb,items_emb)
        scores =  torch.sum(matching,dim=-1)
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
        return bce_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
    
    def predict(self,users,items):
        users  = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.computer()
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb,items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()
    

    def predict_changed_graph(self,users,items,changed_graph):
        users  = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.F_computer(self.embedding_user.weight,self.embedding_item.weight,changed_graph)
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb,items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()




class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(nn.Module):
    def __init__(self, args):
        super(SASRec, self).__init__()
        self.config = args

        self.user_num = args.user_num
        self.item_num = args.item_num
        

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
    
    def _device(self):
        self.dev = self.item_emb.weight.device

    def log2feats(self, log_seqs):
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu().numpy() == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training        
    #     log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     neg_logits = (log_feats * neg_embs).sum(dim=-1)

    #     # pos_pred = self.pos_sigmoid(pos_logits)
    #     # neg_pred = self.neg_sigmoid(neg_logits)

    #     return pos_logits, neg_logits # pos_pred, neg_pred
    def forward_eval(self, user_ids, target_item, log_seqs): # for training
        self._device()        
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        log_feats = log_feats[:,-1,:]
        item_embs = self.item_emb(target_item)
        # pos_embs = self.item_emb(torch.LongTensor(target_item).to(self.dev))
        # neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        return (log_feats*item_embs).sum(dim=-1)

        # pos_logits = (log_feats * pos_embs).sum(dim=-1)
        # neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # # pos_pred = self.pos_sigmoid(pos_logits)
        # # neg_pred = self.neg_sigmoid(neg_logits)

        # return pos_logits, neg_logits # pos_pred, neg_pred
    def forward(self, seqs, target, target_posi=None):
        self._device()    
        # posi_raw = torch.arange(target_posi.shape[0]).unsqueeze(-1).repeat(1,target_posi.shape[1])
        # posi_raw = posi_raw.reshape(-1)
        log_feats = self.log2feats(seqs)
        if target_posi is not None:
            s_emb = log_feats[target_posi[:,0], target_posi[:,1]]
        else:
            s_emb = log_feats[:,-1,:]
        target_embeds = self.item_emb(target.reshape(-1))
        scores = torch.mul(s_emb, target_embeds).sum(dim=-1)
        return scores
    
    def computer(self): # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None
    
    def seq_encoder(self, seqs): # seq embedding server as user embedding for CollabRec
        self._device()
        log_feats = self.log2feats(seqs)
        seq_emb = log_feats[:,-1,:]
        return seq_emb
    
    def item_encoder(self,target_item,all_items=None):
        self._device()
        target_embeds = self.item_emb(target_item)
        return target_embeds
    



    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    
    def predict_all(self, user_ids, log_seqs):
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb.weight

        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits = torch.matmul(final_feat, item_embs.T)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    
    def predict_all_batch(self, user_ids, log_seqs,batch_size=128):
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb.weight

        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits = torch.matmul(final_feat,item_embs.T)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
    
    def log2feats_v2(self, log_seqs, emb_replace=None):
        log_seqs = log_seqs+0
        # if emb is not None:
        emb_replace_idx = np.where(log_seqs<0)
        log_seqs[emb_replace_idx] = 0
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))+0
        log_seqs[emb_replace_idx] = -1
        if emb_replace is not None:
            seqs[emb_replace_idx[0],emb_replace_idx[1]] = 0
            seqs[emb_replace_idx[0],emb_replace_idx[1]] += emb_replace
            # for i in range(emb_replace_idx[0].shape[0]):
            #     # seqs[0,73,:] = emb_replace[i]
            #     seqs[emb_replace_idx[0][i],emb_replace_idx[1][i],:] = emb_replace[i]


        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

    def predict_position(self,log_seqs,postions,emb_replace=None):
        log_feats = self.log2feats_v2(log_seqs,emb_replace=emb_replace) # user_ids hasn't been used yet


        final_feat = log_feats[np.arange(postions.shape[0]), postions] # only use last QKV classifier, a waste

        item_embs = self.item_emb.weight

        # logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits = torch.matmul(final_feat,item_embs.T)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)

#######  DCN modules

class CrossNetwork(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


class DNN(nn.Module):
    """
    This module contains DNN.
    """
    def __init__(self, inputs_dim, hidden_units, activation=nn.ReLU(), use_bn=False,dp=0.2):
        super().__init__()
        # self.dropout = nn.Dropout(keep_prob=1.0-cfg.DROPOUT)
        self.dropout = nn.Dropout(p=dp)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        if inputs_dim > 0:
            hidden_units = [inputs_dim] + list(hidden_units)
        else:
            hidden_units = list(hidden_units)
        
        self.linears = nn.ModuleList([nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units)-1)])
        if self.use_bn:
            self.batch_norm = nn.ModuleList([nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units)-1)])
        self.activation_layers = nn.ModuleList([activation for _ in range(len(hidden_units)-1)])


    def forward(self, inputs):
        deep_input = inputs
        for i, _ in enumerate(self.linears):
            fc_output = self.linears[i](deep_input)
            if self.use_bn:
                fc_output = self.batch_norm[i](fc_output)
            fc_output = self.activation_layers[i](fc_output)
            fc_output = self.dropout(fc_output)
            deep_input = fc_output
        return deep_input


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}
def varlenSparseFeature(feat, feat_num, max_len, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'max_len': max_len, 'embed_dim': embed_dim}

####  DIN

##### DIN modules
class Dice(nn.Module):
    """The Data Adaptive Activation Function in DIN,which can be viewed as a generalization of PReLu and can adaptively adjust the rectified point according to distribution of input data.
    Input shape:
        - 2 dims: [batch_size, embedding_size(features)]
        - 3 dims: [batch_size, num_features, embedding_size(features)]
    Output shape:
        - Same shape as input.
    References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
        - https://github.com/zhougr1993/DeepInterestNetwork, https://github.com/fanoping/DIN-pytorch
    """

    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3

        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        # wrap alpha in nn.Parameter to make it trainable
        if self.dim == 2:
            self.alpha = nn.Parameter(torch.zeros((emb_size,)))
        else:
            self.alpha = nn.Parameter(torch.zeros((emb_size, 1)))

    def forward(self, x):
        assert x.dim() == self.dim
        if self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        return out
    
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_unit, batch_norm=False, activation='relu', sigmoid=False, dropout=None, dice_dim=None):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_unit) >= 1 
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_unit[0]))
        
        for i, h in enumerate(hidden_unit[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_unit[i]))
            
            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activation.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_unit[i], dim=dice_dim))
            else:
                raise NotImplementedError

            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_unit[i], hidden_unit[i+1]))
        
        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x) 

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_unit=[80, 40], embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4*embedding_dim,
                                       hidden_unit=hidden_unit,
                                       batch_norm=batch_norm,
                                       sigmoid=False,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = nn.Linear(hidden_unit[-1], 1)

    # @torchsnooper.snoop()
    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        
        queries = query.expand(-1, user_behavior_len, -1)
        
        attention_input = torch.cat([queries, user_behavior, queries-user_behavior, queries*user_behavior],
             dim=-1) # as the source code, subtraction simulates verctors' difference
        
        attention_output = self.fc1(attention_input)
        attention_score = self.fc2(attention_output) # [B, T, 1]

        return attention_score

class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_unit=[64, 16], embedding_dim=embedding_dim, batch_norm=False)

    
    def forward(self, query_ad, user_behavior, mask=None):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # mask                : size -> batch_size * time_seq_len
        # output              : size -> batch_size * 1 * embedding_size
        
        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask.unsqueeze(1), torch.tensor(0))
        

        # multiply weight
        output = torch.matmul(attention_score, user_behavior)

        return output


class RecEncoder_DIN(nn.Module):
    def __init__(self, args, hidden_units=[200,80,1]):
        super().__init__()
        self.config = args
        self.user_num = int(args.user_num)
        self.item_num = int(args.item_num)
        emb_dim = args.embedding_size//3
        self.sparse_feature_columns = [sparseFeature('uid', self.user_num, embed_dim=emb_dim), sparseFeature('iid', self.item_num, embed_dim=emb_dim)]
        self.sequence_feature_columns = [varlenSparseFeature("his",  self.item_num,
                                                    10, embed_dim=emb_dim)]
        self.layer_num = len(hidden_units)
        emb_dim = self.sparse_feature_columns[0]['embed_dim']
        self.dim = emb_dim * 3 # emb_dim * feat num
        # Creating Embedding layers
        self.embed_layers = nn.ModuleList([nn.Embedding(feat['feat_num'], feat['embed_dim']) for i, feat in enumerate(self.sparse_feature_columns)])
        self.sequence_embed_layers = nn.ModuleList([
            nn.Embedding(feat['feat_num'], feat['embed_dim'])
            for i, feat in enumerate(self.sequence_feature_columns)])
        print("DIN drop our ration:", args.drop)
        self.attn = AttentionSequencePoolingLayer(embedding_dim=emb_dim)
        self.fc_layer = FullyConnectedLayer(input_size=emb_dim*3,
                                            hidden_unit=hidden_units,
                                            batch_norm=False,
                                            sigmoid = True,
                                            activation='dice',
                                            dropout= args.drop,
                                            dice_dim=2)
        # self.act_pre_out = Dice(80,dim=2)
        # self.out_layer = nn.Linear(80,1)
        self._init_weights()
    def _init_weights(self):
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m)
    
    def computer(self):
        return None,None
    
    def user_encoder(self, users,all_users=None):
        user_embs = self.embed_layers[0](users)
        return user_embs
    
    def item_encoder(self,target_item,all_items=None):
        target_embeds =  self.embed_layers[1](target_item)
        target_embeds = torch.cat([target_embeds]*3, dim=-1)
        return target_embeds
    
    def all_encode(self, users,items,seqs):
        # sparse_inputs, sequence_inputs = inputs
        user_emb = self.embed_layers[0](users.squeeze())
        item_emb = self.embed_layers[1](items.squeeze())
        sequence_inputs = (seqs,)
        rec_his_emb = torch.cat(
            [self.sequence_embed_layers[i](sequence_inputs[i].squeeze_(1))
             for i in range(len(self.sequence_feature_columns))],dim=-1)
        rec_his_mask = torch.where(
                            sequence_inputs[0]==0,
                            1, 0).bool()
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb], dim=-1)
        return concat_feature

    def forward(self, inputs):
        sparse_inputs, sequence_inputs = inputs
        # sequence_inputs = P.Split(axis=1)(sequence_inputs)
        # if len(sequence_inputs.shape)<3:
        #     sequence_inputs = sequence_inputs.unsqueeze(1)
        # sequence_inputs = torch.split(sequence_inputs,1,dim=1)
        user_emb = self.embed_layers[0](sparse_inputs[:, 0])
        item_emb = self.embed_layers[1](sparse_inputs[:, 1])
        sequence_inputs = (sequence_inputs,)
        rec_his_emb = torch.cat(
            [self.sequence_embed_layers[i](sequence_inputs[i].squeeze_(1))
             for i in range(len(self.sequence_feature_columns))],dim=-1)
        rec_his_mask = torch.where(
                            sequence_inputs[0]==0,
                            1, 0).bool()
        browse_atten = self.attn(item_emb.unsqueeze(dim=1),
                            rec_his_emb, rec_his_mask) 
        concat_feature = torch.cat([item_emb, browse_atten.squeeze(dim=1), user_emb], dim=-1)

        out = self.fc_layer(concat_feature)

        return out
    