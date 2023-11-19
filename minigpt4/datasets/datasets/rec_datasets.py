import os
from select import select
# from PIL import Image
# import webdataset as wds
from minigpt4.datasets.datasets.rec_base_dataset import RecBaseDataset 
import pandas as pd
import numpy as np
import logging
# from minigpt4.datasets.datasets.caption_datasets import CaptionDataset




# class RecDataset(RecBaseDataset):


#     def __getitem__(self, index):

#         # TODO this assumes image input, not general enough
#         ann = self.annotation.iloc[index]
#         return {
#             "User": ann['User'],
#             "InteractedItems": ann['InteractedItems'],
#             "InteractedItemTitles": ann['InteractedItemTitles'],
#             "TargetItemID": ann["TargetItemID"],
#             "TargetItemTitle": ann["TargetItemTitle"]
#         }
        

def convert_title_list_v2(titles):
    titles_ = []
    for x in titles:
        if len(x)>0:
            titles_.append("\""+ x + "\"")
    if len(titles_)>0:
        return ", ".join(titles_)
    else:
        return "unkow"
def convert_title_list(titles):
    titles = ["\""+ x + "\"" for x in titles]
    return ", ".join(titles)
    

class MovielensDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+".pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid','iid','title','sessionItems', 'sessionItemTitles','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid','iid','title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle','label']
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length = max(max_length,len(x))
            self.max_lenght = max_length
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": ann['InteractedItemTitles'],
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                # "InteractedNum": None,
                "label": ann['label']
            }



class MovielensDataset_stage1(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__(text_processor, ann_paths)
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+".pkl").reset_index(drop=True)[['uid','iid','title','sessionItems', 'sessionItemTitles','label', 'pairItems', 'pairItemTitles']]
        self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label','PairItemIDs','PairItemTitles']
        self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
        self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(convert_title_list)
        self.annotation["PairItemTitles"] = self.annotation["PairItemTitles"].map(convert_title_list)
        
        
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        return {
            "UserID": ann['UserID'],
            "PairItemIDs": np.array(ann['PairItemIDs']),
            "PairItemTitles": ann["PairItemTitles"],
            "label": ann['label']
        }

class AmazonDataset(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+"_seqs.pkl").reset_index(drop=True)
        self.use_his = False
        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            self.use_his = True
            self.annotation = self.annotation[['uid','iid','title','his', 'his_title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            self.annotation = self.annotation[['uid','iid','title','label']]
            self.annotation.columns = ['UserID','TargetItemID','TargetItemTitle','label']
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_,len(x))
            self.max_lenght = min(max_length_, 15) # average: only 5 
            print("amazon datasets, max history length:", self.max_lenght)
            logging.info("amazon datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            return {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"]+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
        else:
            return {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"],
                # "InteractedNum": None,
                "label": ann['label']
            }


class MoiveOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split("=")
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)

        ## warm test:
        if "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['warm'].isin([1])].copy()
        if "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()
        
        
        

        self.use_his = False
        self.prompt_flag = False

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 




class MoiveOOData_sasrec(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None,sas_seq_len=25):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split("=")
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        # self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        
        self.use_his = False
        self.prompt_flag = False
        self.sas_seq_len = sas_seq_len

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            
            if len(a) < self.sas_seq_len: # used for sasrec
                c = [0]*(self.sas_seq_len - len(a))
                c.extend(a)
            elif len(a) >= self.sas_seq_len:
                c = a[-self.sas_seq_len:]

            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "sas_seq": np.array(c)
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 





class AmazonOOData(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        ann_paths = ann_paths[0].split('=') 
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        self.use_his = False
        self.prompt_flag = False

        # ## warm test:
        
        if 'not_cold' in self.annotation.columns and "warm" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([1])].copy()
        if 'not_cold' in self.annotation.columns and "cold" in ann_paths:
            self.annotation = self.annotation[self.annotation['not_cold'].isin([0])].copy()

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 





class AmazonOOData_sasrec(RecBaseDataset):
    def __init__(self, text_processor=None, ann_paths=None,sas_seq_len=20):
        super().__init__()
        # self.vis_root = vis_root
        # self.annotation = pd.read_csv(ann_paths[0],sep='\t', index_col=None,header=0)[['uid','iid','title','sessionItems', 'sessionItemTitles']]
        self.annotation = pd.read_pickle(ann_paths[0]+"_ood2.pkl").reset_index(drop=True)
        
        self.use_his = False
        self.prompt_flag = False
        self.sas_seq_len = sas_seq_len

        if 'sessionItems' in self.annotation.columns or 'his' in self.annotation.columns:
            used_columns = ['uid','iid','title','his', 'his_title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle', 'InteractedItemIDs', 'InteractedItemTitles','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append("not_cold")
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True

            self.use_his = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
            self.annotation["InteractedItemIDs"] = self.annotation["InteractedItemIDs"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"].map(list)
            self.annotation["InteractedItemTitles"] = self.annotation["InteractedItemTitles"] #.map(convert_title_list)
        else:
            used_columns = ['uid','iid','title','label']
            renamed_columns = ['UserID','TargetItemID','TargetItemTitle','label']
            if 'not_cold' in self.annotation.columns:
                used_columns.append('not_cold')
                renamed_columns.append("prompt_flag")
                self.prompt_flag = True
            self.annotation = self.annotation[used_columns]
            self.annotation.columns = renamed_columns
        
        print("data path:", ann_paths[0], "data size:", self.annotation.shape)
        self.user_num = self.annotation['UserID'].max()+1
        self.item_num = self.annotation['TargetItemID'].max()+1
        self.text_processor = text_processor
        
        
        if self.use_his:
            max_length_ = 0
            for x in self.annotation['InteractedItemIDs'].values:
                max_length_ = max(max_length_, len(x))
            self.max_lenght = min(max_length_, 10) # average: only 50; 0915: 15 
            print("Movie OOD datasets, max history length:", self.max_lenght)
            logging.info("Movie OOD datasets, max history length:" + str(self.max_lenght))
            
    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation.iloc[index]
        if self.use_his:
            a = ann['InteractedItemIDs']
            InteractedNum = len(a)
            if a[0] == 0:
                InteractedNum -= 1

            if len(a) < self.max_lenght:
                b = [0]* (self.max_lenght-len(a)) # assuming padding idx is zero
                b.extend(a)
            elif len(a)> self.max_lenght:
                b = a[-self.max_lenght:]
                InteractedNum = self.max_lenght
            else:
                b = a
            
            if len(a) < self.sas_seq_len: # used for sasrec
                c = [0]*(self.sas_seq_len - len(a))
                c.extend(a)
            elif len(a) >= self.sas_seq_len:
                c = a[-self.sas_seq_len:]

            one_sample = {
                "UserID": ann['UserID'],
                "InteractedItemIDs_pad": np.array(b),
                # "InteractedItemIDs": ann['InteractedItemIDs'],
                "InteractedItemTitles": convert_title_list_v2(ann['InteractedItemTitles'][-InteractedNum:]),
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": "\""+ann["TargetItemTitle"].strip(' ')+"\"",
                "InteractedNum": InteractedNum,
                "label": ann['label'],
                "sas_seq": np.array(c)
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 
        else:
            one_sample = {
                "UserID": ann['UserID'],
                # "InteractedItemIDs_pad": None,
                # # "InteractedItemIDs": ann['InteractedItemIDs'],
                # "InteractedItemTitles": None,
                "TargetItemID": ann["TargetItemID"],
                "TargetItemTitle": ann["TargetItemTitle"].strip(' '),
                # "InteractedNum": None,
                "label": ann['label']
            }
            if self.prompt_flag:
                one_sample['prompt_flag'] = ann['prompt_flag']
            return one_sample 


