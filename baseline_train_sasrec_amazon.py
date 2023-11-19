from minigpt4.models.rec_base_models import MatrixFactorization, LightGCN, SASRec 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from minigpt4.datasets.datasets.rec_gnndataset import GnnDataset
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 


import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import time


def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user

class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.embed_size = 64
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = '/home/zyang/code-2022/RecUnlearn/data/'
        self.dataset = 'ml-100k' #'yahoo-s622-01' #'yahoo-small2' #'yahooR3-iid-001'
        self.layer_size='[64,64]'
        self.verbose = 1
        self.Ks='[10]'
        self.data_type='retraining'

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 100
        self.A_split = False
        self.dropout = False
        self.pretrain=0
        self.init_emb=1e-4
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)
    
    def hyper_para_info(self):
        print(self.__dict__)


class seq_dataset_train(Dataset):
    def __init__(self,data_path,max_len=50):
        # super.__init__()
        self.data = pd.read_pickle(data_path)
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_i = self.data.loc[index]
        seqs = data_i.iput_seqs
        targets = data_i.targets
        target_posi = data_i.target_posi
        if len(seqs) < self.max_len:
            padding_len = self.max_len-len(seqs)
            pad_seqs = [0]*padding_len
            pad_seqs.extend(seqs)
            seqs = pad_seqs
            target_posi = np.array(target_posi) + padding_len
        return seqs, targets, target_posi
    
    def batch_generator(self,batch_size):
        idxs = np.arange(self.__len__())
        np.random.shuffle(idxs)
        
        for i_start in range(0,self.__len__(),batch_size):
            i_end = min(self.__len__(), i_start+batch_size)
            sequnces_all = []
            labels_all = []
            targets_all = []
            target_posi_all = []
            raw_id = 0
            for i in range(i_start,i_end):
                data_i = self.data.loc[i]
                seqs = data_i.iput_seqs
                targets = data_i.targets
                target_posi = data_i.target_posi
                labels = data_i.labels
                if len(seqs) < self.max_len:
                    padding_len = self.max_len-len(seqs)
                    pad_seqs = [0] * padding_len
                    pad_seqs.extend(seqs)
                    seqs = pad_seqs
                    target_posi = np.array(target_posi) + padding_len
                    target_posi = [[raw_id,x] for x in target_posi]
                elif len(seqs) > self.max_len:
                    cut_len = len(seqs) - self.max_len
                    seqs = list(np.array(seqs)[-self.max_len:])
                    target_posi = np.array(target_posi) 
                    idxs_used = np.where(target_posi >= cut_len)
                    target_posi = target_posi[idxs_used] - cut_len
                    target_posi = [[raw_id,x] for x in target_posi]
                    labels = np.array(labels)[idxs_used]
                    targets = np.array(targets)[idxs_used]
                else:
                    target_posi = [[raw_id, x] for x in target_posi]

                sequnces_all.append(seqs)
                labels_all.extend(labels)
                targets_all.extend(targets)
                target_posi_all.extend(target_posi)
                # target_posi_all = np.array(target_posi_all)
                raw_id += 1
            yield torch.tensor(sequnces_all), torch.tensor(targets_all),torch.tensor(target_posi_all),torch.tensor(labels_all)


                    







class seq_dataset_eval(Dataset):
    def __init__(self,data,max_len=50):
        # super.__init__()
        self.data = data #pd.read_pickle(data_path).values
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_i = self.data[index]
        uid, iid, his, labels = data_i[0], data_i[1], data_i[2], data_i[3]
        if len(his) < self.max_len:
            his_ = np.zeros(self.max_len)
            his_[-len(his):] = np.array(his)
            his = his_
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his) 
        return  uid, iid, his, labels       


        # 'iput_seqs','targets','target_posi' 
        # data_i = self.data[index]
        # uid,iid,his,labels = data_i[0], data_i[1], data_i[2], data_i[3]
        # if len(his) < self.max_len:
        #     his = np.zeros(self.max_len)
        # return  uid, iid, his, labels   




class seq_dataset(Dataset):
    def __init__(self,data,max_len=10):
        # super.__init__()
        self.data = data
        self.max_len = max_len
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data_i = self.data[index]
        uid, iid, his, labels = data_i[0], data_i[1], data_i[2], data_i[3]
        if len(his) < self.max_len:
            his_ = np.zeros(self.max_len)
            his_[-len(his):] = np.array(his)
            his = his_
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his) 
        return  uid, iid, his, labels     

class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True):
    seed=2023
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # args = model_hyparameters()
    # args.reset(train_config)
    # args.hyper_para_info()

    # load dataset
    # data_dir = "/home/zyang/LLM/MiniGPT-4/dataset/ml-100k/"
    # data_dir = "/home/sist/zyang/LLM/datasets/ml-100k/"
    # train_data = pd.read_pickle(data_dir+"train.pkl")[['uid','iid',"sessionItems",'label']].values
    # valid_data = pd.read_pickle(data_dir+"valid.pkl")[['uid','iid',"sessionItems",'label']].values
    # test_data = pd.read_pickle(data_dir+"test.pkl")[['uid','iid',"sessionItems",'label']].values

    # train_config={
    #     "lr": 1e-2,
    #     "wd": 1e-4,
    #     "epoch": 5000,
    #     "eval_epoch":1,
    #     "patience":50,
    #     "batch_size":1024
    # }

    data_dir = "/data/zyang/datasets/amazon_book_new/"
    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid', 'his', 'label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid', 'his', 'label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid', 'his', 'label']].values

    user_num = max(train_data[:,0].max(),valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num = max(train_data[:,1].max(),valid_data[:,1].max(), test_data[:,1].max()) + 1

    train_data = seq_dataset(train_data,max_len=int(train_config['maxlen']))
    valid_data = seq_dataset_eval(valid_data,max_len=int(train_config['maxlen']))
    test_data = seq_dataset_eval(test_data,max_len=int(train_config['maxlen']))



    sasrec_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "hidden_units": int(embedding_size),
        "num_blocks": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "l2_emb": 1e-4,
        "maxlen": int(train_config['maxlen'])
        }
    print("sasrec_config:\n", sasrec_config)
    sasrec_config = omegaconf.OmegaConf.create(sasrec_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)





    # model = MatrixFactorization(mf_config).cuda()
    # model = LightGCN(lgcn_config).cuda()
    model = SASRec(sasrec_config).cuda()
    # model._set_graph(gnndata.Graph)
    
    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'], weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = [x_.cuda() for x_ in batch_data]
            ui_matching = model.forward_eval(batch_data[0].long(),batch_data[1].long(),batch_data[2].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[-1].cpu().numpy())
            users.extend(batch_data[0].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc = 0
        valid_uauc, _, _ = uAUC_me(users,pre,label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = [x_.cuda() for x_ in batch_data]
            ui_matching = model.forward_eval(batch_data[0].long().cuda(),batch_data[1].long().cuda(),batch_data[2].long().cuda())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[-1].cpu().numpy())
            users.extend(batch_data[0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        # test_uauc = 0
        test_uauc, _, _ = uAUC_me(users,pre,label)
        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc: {} acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return 
    

    for epoch in range(train_config['epoch']):
        model.train()
        for bacth_id, batch_data in enumerate(train_data_loader):
            batch_data = [x_.cuda() for x_ in batch_data]
            ui_matching = model(batch_data[2].long(), batch_data[1].long()) # seqs, targets
            loss = criterion(ui_matching, batch_data[-1].float().reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id, batch_data in enumerate(valid_data_loader):
                try:
                    batch_data = [x_.cuda() for x_ in batch_data]
                except:
                    pass
                ui_matching = model.forward_eval(batch_data[0].long(),batch_data[1].long(),batch_data[2].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[-1].cpu().numpy())
                users.extend(batch_data[0].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc = 0
            # valid_uauc, _, _ = uAUC_me(users,pre,label)

            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = [x_.cuda() for x_ in batch_data]
                ui_matching = model.forward_eval(batch_data[0].long(),batch_data[1].long(),batch_data[2].long())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[-1].cpu().numpy())
                users.extend(batch_data[0].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            test_uauc = 0
            # test_uauc, _, _ = uAUC_me(users,pre,label)
            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc, 'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric) 
    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()

# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-4]
#     dw_ = [1e-1, 1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,0]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64,128,256]
#     max_len = 20
    
#     try:
#         f = open("0925book-sasrec_head1_search_lr"+str(lr_[0])+"len"+str(max_len)+".log",'rw+')
#         # f = open("ml100k-sasrec_search_lrall-int0.1_p100_1layer"+".log",'rw+')
#     except:
#         f = open("0925book-sasrec_head1_search_lr"+str(lr_[0])+"len"+str(max_len)+".log",'w+')
#         # f = open("ml100k-sasrec_lgcn_search_lrall-int0.1_p100_1layer"+".log",'w+')
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size": 2048*5, #2048,
#                     "maxlen": max_len
#                 }
#                 print(train_config)
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=False)
#     f.close()


# {'lr': 0.01, 'wd': 0.01, 'embedding_size': 64, 'epoch': 5000, 'eval_epoch': 1, 'patience': 100, 
#  'batch_size': 2048, 'maxlen': 25}, {'valid_auc': 0.6901948441436031, 'valid_uauc': 0.681306392663344, 
# 'test_auc': 0.7078362163379921, 'test_uauc': 0.6738139006659691, 'epoch': 194}) 
# save version....
# if __name__=='__main__':
#     # lr_ = [1e-1,1e-2,1e-3]
#     lr_=[1e-3] #1e-2
#     dw_ = [1e-3]
#     # embedding_size_ = [32, 64, 128, 156, 512]
#     embedding_size_ = [64]
#     save_path = "/data/zyang/LLM/PretrainedModels/sasrec/"
#     # save_path = "/home/sist/zyang/LLM/PretrainedModels/mf/"
#     # try:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
#     # except:
#     #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
#     f=None
#     for lr in lr_:
#         for wd in dw_:
#             for embedding_size in embedding_size_:
#                 train_config={
#                     'lr': lr,
#                     'wd': wd,
#                     'embedding_size': embedding_size,
#                     "epoch": 5000,
#                     "eval_epoch":1,
#                     "patience":100,
#                     "batch_size": 2048*5, #2048,
#                     "maxlen": 20
#                 }
#                 print(train_config)
#                 save_path += "0928sasrec_book_oodv2_best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) +"len"+str(train_config['maxlen']) + ".pth"
#                 run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path)
#     f.close()


# with prtrain version:
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-2] #1e-2
    dw_ = [1e-4]
    # embedding_size_ = [32, 64, 128, 156, 512]
    embedding_size_ = [64]
    save_path = "/data/zyang/LLM/PretrainedModels/sasrec/"
    # try:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'rw+')
    # except:
    #     f = open("rec_mf_search_lr"+str(lr_[0])+".log",'w+')
    f=None
    for lr in lr_:
        for wd in dw_:
            for embedding_size in embedding_size_:
                train_config={
                    'lr': lr,
                    'wd': wd,
                    'embedding_size': embedding_size,
                    "epoch": 5000,
                    "eval_epoch":1,
                    "patience":50,
                    "batch_size":10240,
                    "maxlen": 20
                }
                print(train_config)
                # if os.path.exists(save_path + "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"):
                #     save_path += "best_model_d" + str(embedding_size)+ 'lr-'+ str(lr) + "wd"+str(wd) + ".pth"
                # else:
                #     save_path += "best_model_d" + str(embedding_size) + ".pth"
                save_path = save_path + "0928sasrec_book_oodv2_best_model_d64lr-0.001wd0.001len20_true.pth"
                
                run_a_trail(train_config=train_config, log_file=f, save_mode=True,save_file=save_path,need_train=False)
    if f is not None:
        f.close()
        





        
            







