import numpy as np
import pandas as pd
import ast

def find_max(log_path):
    with open(log_path,'r') as f:
        lines = f.readlines()
        configs = []
        results = []
        for  one_line in lines:
            # print(one_line)
            one_line = one_line.strip("train_config: ").split(" best result: ")
            try:
                configs.append(ast.literal_eval(one_line[0]))
                results.append(ast.literal_eval(one_line[1]))
            except:
                print(one_line)
                raise RuntimeError

    max_k = 0
    max_valid_auc = 0
    for k in range(len(results)):
        if results[k]['valid_auc'] > max_valid_auc:
            max_k = k
            max_valid_auc = results[k]['valid_auc']
    return configs[max_k], results[max_k]

#### ml-100k
# # print(find_max("rec_mf_search_lr0.1.log"),'\n')
# # print(find_max("rec_mf_search_lr0.01.log"),'\n')
# # print(find_max("rec_mf_search_lr0.001.log"),'\n')
# # print(find_max("rec_mf_search_lr0.0001.log"))
# print(find_max("ml100k-rec_lgcn_search_lr0.01.log"),'\n')
# print(find_max("ml100k-rec_lgcn_search_lr0.001.log"),'\n')
# print(find_max("ml100k-rec_lgcn_search_lr0.0001.log"),'\n')


# print("init 0.001")
# print(find_max("ml100k-rec_lgcn_search_lrall-int0.001.log"),'\n')
# print("init 0.01")
# print(find_max("ml100k-rec_lgcn_search_lrall-int0.01.log"),'\n')
# print("init 0.1")
# print(find_max("ml100k-rec_lgcn_search_lrall-int0.1.log"),'\n')

# print("init 0.1, patience:100")
# print(find_max("ml100k-rec_lgcn_search_lrall-int0.1_p100.log"),'\n')

# print("init 0.1, patience:100, 1 layer")
# print(find_max("ml100k-rec_lgcn_search_lrall-int0.1_p100_1layer.log"),'\n')

# print("init 0.1, patience:100, 2 layer")
# print(find_max("ml100k-rec_lgcn_search_lrall-int0.1_p100_2layer.log"),'\n')



#### amazon 
# print(find_max("0827amzon-rec_mf_search_lr0.01.log"),'\n')
# print(find_max("0827amzon-rec_mf_search_lr0.001.log"),'\n')
# print(find_max("0827amzon-rec_mf_search_lr0.0001.log"),'\n')




# print(find_max("rec_mf_search_lr0.01.log"),'\n')
# print(find_max("rec_mf_search_lr0.001.log"),'\n')
# print(find_max("rec_mf_search_lr0.0001.log"))



# ### ood ml-100 lightgcn 

# print(find_max("ood-ml100k-rec_lgcn_search_lrall-int0.1_p100_1layer0.001.log"),'\n')
# print(find_max("ood-ml100k-rec_lgcn_search_lrall-int0.1_p100_1layer0.01.log"),'\n')


# ### ood v2 ml-100 mf:
# print(find_max("0912ml100-ood-v2-rec_mf_search_lr0.1.log"),'\n')
# print(find_max("0912ml100-ood-v2-rec_mf_search_lr0.01.log"),'\n')
# print(find_max("0912ml100-ood-v2-rec_mf_search_lr0.001.log"),'\n')
# print(find_max("0912ml100-ood-v2-rec_mf_search_lr0.0001.log"),'\n')



# ## ood v2 ml-1m mf:
# print(find_max("0913ml1m-ood-v2-rec_mf_search_lr0.1.log"),'\n')
# print(find_max("0913ml1m-ood-v2-rec_mf_search_lr0.01.log"),'\n')
# print(find_max("0913ml1m-ood-v2-rec_mf_search_lr0.001.log"),'\n')
# print(find_max("0913ml1m-ood-v2-rec_mf_search_lr0.0001.log"),'\n')

# ## ood v2 ml-1m lgcm:
# print(find_max("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer0.01.log"),'\n')
# print(find_max("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer0.001.log"),'\n')
# print(find_max("0919-oodv2-ml1m-rec_lgcn_search_lrall-int0.1_p100_1layer0.0001.log"),'\n')


# ### ood v2 ml-1m sasrec:
# print(find_max("0919ml1m-sasrec_lgcn_search_lr0.01len25.log"),'\n')
# print(find_max("0919ml1m-sasrec_lgcn_search_lr0.001len25.log"),'\n')
# print(find_max("0919ml1m-sasrec_lgcn_search_lr0.0001len25.log"),'\n')



# ### ood v2 book lightgcn:
# print(find_max("0924-oodv2-book-lgcn_search_lrall-int0.1_p100_0.01.log"),'\n')
# print(find_max("0924-oodv2-book-lgcn_search_lrall-int0.1_p100_0.001.log"),'\n')
# print(find_max("0924-oodv2-book-lgcn_search_lrall-int0.1_p100_0.0001.log"),'\n')




# ood v2 book sasrec
print(find_max("0925book-sasrec_head1_search_lr0.01len20.log"),'\n')
print(find_max("0925book-sasrec_head1_search_lr0.001len20.log"),'\n')
print(find_max("0925book-sasrec_head1_search_lr0.0001len20.log"),'\n')


