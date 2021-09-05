#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gc
import networkx as nx
import time
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
pd.set_option('display.max_columns',None)

ROOT_PATH = '../data/'
FEAT_PATH = './feature/'


# In[2]:


def to_str(x):
    if isinstance(x,str):
#         print(x)
        xlist = x.split(':')
        return xlist[1]
    else:
        return x

def prepare(train_data, test_data):
    train_data = train_data.applymap(to_str)
    test_data = test_data.applymap(to_str)

    train_data.columns = ['click', 'resume'] + [str(i) for i in range(1, 90) if i not in [4,8,9,10,11,12,14,15,17,19,42,44,49,50,51,53,56,57,85,86]]
    test_data.columns = [str(i) for i in range(1, 90) if i not in [4,5,6,8,9,10,11,12,14,15,17,19,42,44,49,50,51,53,56,57,85,86]]

    train_data['resume'] = train_data[['click', 'resume']].apply(lambda x: -1 if int(x[0] == 0) else 1 if int(x[1] == 1) else 0, axis=1)

    data_all = pd.concat([train_data, test_data]).reset_index(drop=True)

    data_all.drop(['18'], axis=1, inplace=True)
    feat = [i for i in train_data.columns if i not in ['click', 'resume'] + ['5','6','7','18','47','48','88','89']]
    data_all[[i for i in feat if i not in ['58','59','60','61']]] = data_all[[i for i in feat if i not in ['58','59','60','61']]].applymap(lambda x: int(x))
    data_all[['58','59','60','61']] = data_all[['58','59','60','61']].applymap(lambda x: float(x))

    for i in ['7','47','48','88']:
        data_all[i] = data_all[i].apply(lambda x: x.split(','))

    train_data = data_all[~data_all['click'].isna()]
    test_data = data_all[data_all['click'].isna()]
    
    return train_data, test_data

print("读取文件：")
train_data = pd.read_csv(ROOT_PATH + 'train.txt', sep='\t', header=None)
test_data = pd.read_csv(ROOT_PATH + 'test.txt', sep='\t', header=None)

print("数据预处理：")
train_data, test_data = prepare(train_data, test_data)


# In[3]:


def neg_data(train_data, test_data):
    
    train_data = train_data.drop(train_data[train_data['1']>900].index)
    
    train_neg = train_data[train_data['click']==0]
    train_neg = train_neg.sample(frac=0.5, random_state=42, replace=False)
    train_data = pd.concat([train_neg, train_data[train_data['click'] == 1]])
    
    data_all = pd.concat([train_data, test_data]).reset_index(drop=True)
    
    return data_all

print("下采样")
data_all = neg_data(train_data, test_data)
data_all.to_feather(FEAT_PATH + 'data_all_prepare_2.feather')


# # 特征提取

# In[2]:


# data_all = pd.read_feather(FEAT_PATH + 'data_all_prepare.feather')
data_all = data_all.reset_index().rename(columns={'index':'uid'}).replace(-1, np.nan)

def get_feat(data_all):
    
    # 序列长度特征
    for i in tqdm(['7','47','48']):
        data_all[ i + '_len'] = data_all[i].apply(lambda x: len(x))
    
    # 福利特征
    pivot = data_all[['{}'.format(i) for i in range(23,36)]].copy()
    data_all['welfare'] = pivot.sum(axis=1)
    
    # 点击投递特征
    data_all['37_38_sub'] = data_all['37'] - data_all['38']
    data_all['80_81_sub'] = data_all['80'] - data_all['81']
    data_all['click_rate'] = data_all['87'] / data_all['83']
    data_all['resume_rate'] = data_all['84'] / data_all['83']
    
    # 职位类别特征
    for key in ['45', '46']:

        group = data_all.groupby(key)['uid'].agg('count').reset_index()
        group.columns = [key, key+'_uid_count']
        data_all = pd.merge(data_all, group, on=key, how='left')

        group = data_all.groupby(key)['16'].agg('nunique').reset_index()
        group.columns = [key, key+'_16_nunique']
        data_all = pd.merge(data_all, group, on=key, how='left')
    
    key = '16'

    #该职位数据条数
    group = data_all.groupby(key)['uid'].agg('count').reset_index()
    group.columns = [key, key+'_uid_count']
    data_all = pd.merge(data_all, group, on=key, how='left')

    agg_list = ['mean','sum']
    group = data_all.groupby(key)['2'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_2_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 用户对该职位访问时刻特征
    agg_list = ['mean', 'max', 'min', 'median', 'nunique']
    group = data_all[data_all['3']!=-1].groupby(key)['3'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_3_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户曝光量特征
    group = data_all[data_all['54']!=-1].groupby(key)['54'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_54_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户投递量特征
    group = data_all[data_all['55']!=-1].groupby(key)['55'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_55_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    agg_list = ['mean','median']
    # 该职位用户工作经验特征
    group = data_all[data_all['62']!=-1].groupby(key)['62'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_62_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户期望薪资特征
    group = data_all[data_all['63']!=-1].groupby(key)['63'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_63_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户期望教育水平特征
    group = data_all[data_all['64']!=-1].groupby(key)['64'].agg(agg_list).reset_index()
    group.columns = [key] + [(key+'_64_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    agg_list = ['max', 'min', 'mean']
    # 该职位用户简历添加时间特征
    group = data_all[data_all['80']!=-1].groupby(key)['80'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_80_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户简历更新时间特征
    group = data_all[data_all['81']!=-1].groupby(key)['81'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_81_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户简历完整度特征
    group = data_all[data_all['82']!=-1].groupby(key)['82'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_82_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    agg_list = ['max', 'min', 'mean', 'median', 'sum']

    # 该职位用户曝光量特征
    group = data_all[data_all['83']!=-1].groupby(key)['83'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_83_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户投递量特征
    group = data_all[data_all['84']!=-1].groupby(key)['84'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_84_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    # 该职位用户点击量特征
    group = data_all[data_all['87']!=-1].groupby(key)['87'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_87_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    group = data_all[data_all['click_rate']!=-1].groupby(key)['click_rate'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_click_rate_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')

    group = data_all[data_all['resume_rate']!=-1].groupby(key)['resume_rate'].agg(agg_list).reset_index()
    group.columns = [key] + [(key + '_resume_rate_{}').format(i) for i in agg_list]
    data_all = pd.merge(data_all, group, on=key, how='left')
    
    return data_all

print("特征构建")
print("FEAT 1")
data_all = get_feat(data_all)
data_all = data_all.fillna(-1)
data_all.drop(['5','6','7','47','48','88','89'], axis=1, inplace=True)
data_all.to_feather(FEAT_PATH + 'data_all_2.feather')


# In[5]:


del train_data
del test_data
del data_all

gc.collect()


# ## 序列特征构造

# In[2]:


def get_vec(data, tag, emb_size):
    sentences = data[tag].values.tolist()
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]

    model = Word2Vec(sentences, size=emb_size, window=10, min_count=5, sg=1, hs=1, seed=1, iter=5, workers=5)

    emb_matrix = []
    for seq in tqdm(sentences):
        vec = []
        for w in seq:
            if w in model.wv.vocab:
                vec.append(model.wv[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    
    for i in range(emb_size):
        data['{}_emb_{}'.format(tag, i)] = emb_matrix[:, i]
        
    return data


# In[3]:


data_all = pd.read_feather(FEAT_PATH + 'data_all_prepare_2.feather')

def get_feat2(data):
    
    data_all = data.copy()
    
    pivot = data_all[['7','47','48','88']]
    
    for tag in ['47','48']:
        pivot = get_vec(pivot, tag, emb_size=8)
    for tag in ['7','88']:
        pivot = get_vec(pivot, tag, emb_size=96)
    
    pivot.drop(['7','47','48','88'], axis=1, inplace=True)
    pivot.to_feather(FEAT_PATH + 'data_feat1_2.feather')

print("FEAT 2")
get_feat2(data_all)


# In[4]:


# data_all = pd.read_feather(FEAT_PATH + 'data_all_prepare.feather')

def get_feat3(data):
    
    data_all = data.copy()
    
    data_all['89'] = data_all['89'].apply(lambda x: [i.split(';') for i in x.split(',')])
    
    pivot = data_all['89'].reset_index()
    pivot.columns = ['uid', 'val_89']

#     pivot.to_feather(FEAT_PATH + 'test_pivot1.feather')

    ind = []
    val = []

    for sub_data in pivot.values:
        s_val = sub_data[1]
        for x in s_val:
            ind.append(sub_data[0])
            val.append(x)

    new_data = pd.DataFrame()

    new_data['uid'] = ind
    new_data['val_89'] = val

    for tag in ['val_89']:
        new_data = get_vec(new_data, tag, emb_size = 64)

    new_data.drop(['val_89'], axis=1, inplace=True)
    new_data.to_feather(FEAT_PATH + 'data_feat2_2.feather')

print("FEAT 3")
get_feat3(data_all)


# In[5]:


# data_all = pd.read_feather(FEAT_PATH + 'data_all_prepare.feather')

def get_feat4(data):
    
    data_all = data.copy()
    
    # pivot = pd.concat([data_all['16'],data_all['88']]).reset_index()
    pivot = data_all[['88','7']].reset_index()
    pivot.columns = ['uid','val_88','val_7']
    
    pivot.to_feather(FEAT_PATH + 'pivot_2.feather')

    data_feat = pivot[['uid']]
    
    uid = []
    val_88 = []
    val_7 = []

    for sub_data in pivot.values:
        s_val = sub_data[1]
        for x in s_val:
            uid.append(sub_data[0])
            val_88.append(x)
            val_7.append(sub_data[2])

    new_data = pd.DataFrame()

    new_data['uid'] = uid
    new_data['val_88'] = val_88
    new_data['val_7'] =val_7
    
    for tag in ['val_7']:
        new_data = get_vec(new_data, tag, emb_size = 32)
        
    new_data.to_feather(FEAT_PATH+'new_data_2.feather')

print("FEAT 4")
get_feat4(data_all)

del data_all
gc.collect()


# In[6]:


def get_feat5(new_data):
    
    pivot = pd.read_feather(FEAT_PATH + 'pivot_2.feather')
    data_feat = pivot[['uid']]
    
    group = new_data[new_data['val_88']!=-1].groupby(['val_88'])['uid'].agg('nunique').reset_index()
    group.columns = ['val_88','val_88_uid_nunique']
    new_data = pd.merge(new_data, group, on='val_88', how='left')

    group = new_data[new_data['val_88']!=-1].groupby(['val_88'])['uid'].agg('count').reset_index()
    group.columns = ['val_88','val_88_uid_count']
    new_data = pd.merge(new_data, group, on='val_88', how='left')
    
    group = new_data[new_data['val_88']!=-1].groupby(['uid'])['val_88'].agg('nunique').reset_index()
    group.columns = ['uid','uid_val_88_nunique']
    new_data = pd.merge(new_data, group, on='uid', how='left')

    group = new_data[new_data['val_88']!=-1].groupby(['uid'])['val_88'].agg('count').reset_index()
    group.columns = ['uid','uid_val_88_count']
    new_data = pd.merge(new_data, group, on='uid', how='left')
    
    for i in tqdm(range(0, 32)):
        group = new_data[new_data['val_88']!=-1].groupby(['val_88'])['val_7_emb_{}'.format(i)].agg('mean').reset_index()
        group.columns = ['val_88', 'val_7_emb_{}_mean'.format(i)]
        new_data = pd.merge(new_data, group, on='val_88', how='left')
        
    
    feats = ['val_88_uid_nunique', 'val_88_uid_count']
    agg_list = ['mean','median', 'max', 'min', 'sum']
    key = 'uid'
    
    for feat in feats:
        group = new_data.groupby(key)[feat].agg(agg_list).reset_index()
        group.columns = [key] + [(key + '_'+ feat +'_{}').format(i) for i in agg_list]
        data_feat = pd.merge(data_feat, group, on=key, how='left')
    
    
    feats = ['uid_val_88_nunique', 'uid_val_88_count']
    agg_list = ['mean']
    key = 'uid'
    
    for feat in feats:
        group = new_data.groupby(key)[feat].agg(agg_list).reset_index()
        group.columns = [key] + [(key + '_'+ feat +'_{}').format(i) for i in agg_list]
        data_feat = pd.merge(data_feat, group, on=key, how='left')
        
    
    feats = ['val_7_emb_{}_mean'.format(i) for i in range(0,32)]
    agg_list = ['mean']
    key = 'uid'
    
    for feat in tqdm(feats):
        group = new_data.groupby(key)[feat].agg(agg_list).reset_index()
        group.columns = [key] + [(key + '_'+ feat +'_{}').format(i) for i in agg_list]
        data_feat = pd.merge(data_feat, group, on=key, how='left')
    
    data_feat.to_feather(FEAT_PATH + 'data_feat3_2.feather')

new_data = pd.read_feather(FEAT_PATH+'new_data_2.feather')
get_feat5(new_data)


# In[7]:


del new_data
gc.collect()


# # 职位特征拼接

# In[8]:


def data_merge():
    
    data_all = pd.read_feather(FEAT_PATH + 'data_all_2.feather')
    data_feat = pd.read_feather(FEAT_PATH + 'data_feat1_2.feather')
    data_all = pd.concat([data_all, data_feat], axis=1)

    data_feat = pd.read_feather(FEAT_PATH + 'data_feat2_2.feather')

    group1 = data_feat.groupby(['uid'])[['val_89_emb_{}'.format(i) for i in range(0,64)]].mean()
    group1.columns = ['val_89_emb_{}_mean'.format(i) for i in range(0,64)]

    group2 = data_feat.groupby(['uid'])[['val_89_emb_{}'.format(i) for i in range(0,64)]].max()
    group2.columns = ['val_89_emb_{}_max'.format(i) for i in range(0,64)]

    group3 = data_feat.groupby(['uid'])[['val_89_emb_{}'.format(i) for i in range(0,64)]].min()
    group3.columns = ['val_89_emb_{}_min'.format(i) for i in range(0,64)]

    data_all = pd.merge(data_all, group1, on='uid', how='left')
    data_all = pd.merge(data_all, group2, on='uid', how='left')
    data_all = pd.merge(data_all, group3, on='uid', how='left')

    data_feat = pd.read_feather(FEAT_PATH + 'data_feat3_2.feather')

    data_all = pd.merge(data_all, data_feat, on='uid', how='left')

    train_data = data_all[data_all['click']!=-1].reset_index(drop=True)
    test_data = data_all[data_all['click']==-1].reset_index(drop=True)

    train_data.to_feather('train_data_2.feather')
    test_data.to_feather('test_data_2.feather')

print("特征拼接")
data_merge()


# # 模型训练

# In[3]:


from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import  CatBoostClassifier
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV


# In[3]:


def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb'):
    
    ### lgb
    if m_type == 'lgb':
        
        params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 64,
                'max_depth':7,
                'learning_rate': 0.02,
                'feature_fraction': 0.85,
                'feature_fraction_seed':2021,
                'bagging_fraction': 0.85,
                'bagging_freq': 5,
                'bagging_seed':2021,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.5,
                'lambda_l2': 1.2,
                'verbose': -1
            }
        
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)
        model = lgb.train(
            params,
            train_set = dtrain,
            num_boost_round=1000,
            valid_sets = [dtrain, dvalid],
            verbose_eval=100,
#             early_stopping_rounds=50,
#             categorical_feature=cat_cols,
        )
    
    ### xgb
    elif m_type == 'xgb':
        
        params = {'booster':'gbtree', #线性模型效果不如树模型
              'objective':'binary:logistic',
              'eval_metric':'auc',
              'eta':0.01, #学习率典型值为0.01-0.2
              'max_depth':7, #树最大深度，典型值为3-10，用来避免过拟合
              'min_child_weight':5, #默认取1，用于避免过拟合，参数过大会导致欠拟合
              'gamma':0.2, #默认取0，该参数指定了节点分裂所需的小损失函数下降值
              'lambda':1, #默认取1.权重的L2正则化项
              'colsample_bylevel':0.7,
              'colsample_bytree':0.8, #默认取1，典型值0.5-1，用来控制每棵树随机采样的比例
              'subsample':0.8, #默认取1，典型值0.5-1，用来控制对于每棵树，随机采样的比例
              'scale_pos_weight':1, #在各类样本十分不平衡时，设定该参数为一个正值，可使算法更快收敛
              }
        
        dtrain = xgb.DMatrix(train_x, label = train_y)
        dvalid = xgb.DMatrix(valid_x, label = valid_y) #测试集特征
        #训练,5167次
        watchlist = [(dtrain,'train'),(dvalid,'valid')] #列出每次迭代的结果
        model = xgb.train(params,dtrain,num_boost_round = 1200,evals = watchlist, verbose_eval=100)   
    
    
    ### cat
    elif m_type == 'cat':
        
        model = CatBoostClassifier(
                 iterations=500,
#                  od_type='Iter',
#                  od_wait=50,
                 max_depth=10,
                 learning_rate=0.07,
                 l2_leaf_reg=9,
                 random_seed=2018,
                 fold_len_multiplier=1.1,
#                  early_stopping_rounds=50,
                 use_best_model=True,
                 loss_function='Logloss',
                 eval_metric='AUC',
                 verbose=100)
        
        model.fit(train_x,train_y,eval_set=[(train_x, train_y),(valid_x, valid_y)], plot=True)   

    return model


# In[2]:


train_data = pd.read_feather('train_data.feather')
test_data = pd.read_feather('test_data.feather')


# In[3]:


train_data


# In[4]:


def train_model():
    
    train_data = pd.read_feather('train_data.feather')
    test_data = pd.read_feather('test_data.feather')
    
    feature = [i for i in train_data.columns if i not in ['uid','click', 'resume' , '5', '6' , '7', '47', '48', '88', '89']]
    lab_col = ['click','resume']
    print(feature)
    print(len(feature))
    
    for m_type in ['lgb', 'cat', 'xgb']:

        train_start = time.time()
        pre_test = pd.DataFrame()
        feat_imp_df = pd.DataFrame({'feat': feature, 'click_imp': 0, 'resume_imp': 0})
        test = test_data[feature]

        valid_result = {}
        for label in lab_col:

            j=0

            train_x = train_data[train_data[label]!=-1][feature].reset_index(drop=True)
            train_y = train_data[train_data[label]!=-1][label].reset_index(drop=True)

            pre_valid = np.zeros((train_x.shape[0],1))

            #在处理后，每个样本已经独立，可以进行k折交叉验证
            seeds = [2021]
            for model_seed in range(len(seeds)):
                print("====================seeds{}====================".format(model_seed))
                kf = StratifiedKFold(n_splits=10, random_state=seeds[model_seed], shuffle=True)
                for train_index, valid_index in kf.split(train_x,train_y):

                    start = time.time()
                    print("************************"+m_type+":{}******************************".format(j+1))
                    #v_train是本次训练的训练集
                    #v_test是本次训练的测试集
                    v_train_x,v_train_y =train_x.loc[train_index], train_y[train_index]
                    v_valid_x,v_valid_y =train_x.loc[valid_index], train_y[valid_index]

                    #通过关键字m_type调用相应模型
                    model = get_model_type(v_train_x,v_train_y,v_valid_x,v_valid_y,m_type=m_type)

                    if m_type=='cat':
                        t = model.predict_proba(v_valid_x)[:,1]
                        pre_valid[valid_index] = t.reshape(-1,1)
                        scoret = model.predict_proba(test)[:,1]
                        pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
                    elif m_type=='xgb':
                        t = model.predict(xgb.DMatrix(v_valid_x))
                        pre_valid[valid_index] = t.reshape(-1,1)
                        scoret = model.predict(xgb.DMatrix(test)) 
                        pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
                    elif m_type=='lgb':
                        feat_imp_df['{}_imp'.format(label)] += model.feature_importance() / 5
                        t = model.predict(v_valid_x)
                        pre_valid[valid_index] = t.reshape(-1,1)
                        scoret = model.predict(test) 
                        pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
                    else:
                        t = model.predict_proba(v_valid_x)[:,1]
                        pre_valid[valid_index] = t.reshape(-1,1)
                        scoret = model.predict_proba(test)[:,1]
                        pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret

                    j = j+1

                    end = time.time()
                    print('runtime: ' + str(end-start))
                valid_result[label] = pre_valid

                print(pre_valid[:5])
                print("AUC score: {}".format(roc_auc_score(train_y, pre_valid)))
                print("ACC score: {}".format(accuracy_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))
                print("F1 score: {}".format(f1_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))
                print("Precision score: {}".format(precision_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))
                print("Recall score: {}".format(recall_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))

        train_end = time.time()
        print('runtime_all: ' + str(train_end - train_start))

        model_click = pd.DataFrame(valid_result['click'])
        model_click.columns = ['{}_click'.format(m_type)]
        model_click.to_feather('./result/{}_click.feather'.format(m_type))

        model_resume = pd.DataFrame(valid_result['resume'])
        model_resume.columns = ['{}_resume'.format(m_type)]
        model_resume.to_feather('./result/{}_resume.feather'.format(m_type))

        pre_test.to_feather('./result/{}_test.feather'.format(m_type))

print("模型训练")
train_model()


# In[12]:


lgb_click = pd.read_feather('./result/lgb_click.feather')
cat_click = pd.read_feather('./result/cat_click.feather')
xgb_click = pd.read_feather('./result/xgb_click.feather')

lgb_resume = pd.read_feather('./result/lgb_resume.feather')
cat_resume = pd.read_feather('./result/cat_resume.feather')
xgb_resume = pd.read_feather('./result/xgb_resume.feather')

lgb_test = pd.read_feather('./result/lgb_test.feather')
cat_test = pd.read_feather('./result/cat_test.feather')
xgb_test = pd.read_feather('./result/xgb_test.feather')

train_click = pd.DataFrame()
train_click['lgb_click'] = lgb_click['lgb_click']
train_click['cat_click'] = cat_click['cat_click']
train_click['xgb_click'] = xgb_click['xgb_click']

train_resume = pd.DataFrame()
train_resume['lgb_resume'] = lgb_resume['lgb_resume']
train_resume['cat_resume'] = cat_resume['cat_resume']
train_resume['xgb_resume'] = xgb_resume['xgb_resume']

test_click = pd.DataFrame()
test_click['lgb_click'] = lgb_test[['pre{}_{}_{}'.format('click', 0, j) for j in range(0, 10)]].sum(axis=1) / 10
test_click['cat_click'] = cat_test[['pre{}_{}_{}'.format('click', 0, j) for j in range(0, 10)]].sum(axis=1) / 10
test_click['xgb_click'] = xgb_test[['pre{}_{}_{}'.format('click', 0, j) for j in range(0, 10)]].sum(axis=1) / 10

test_resume = pd.DataFrame()
test_resume['lgb_resume'] = lgb_test[['pre{}_{}_{}'.format('resume', 0, j) for j in range(0, 10)]].sum(axis=1) / 10
test_resume['cat_resume'] = cat_test[['pre{}_{}_{}'.format('resume', 0, j) for j in range(0, 10)]].sum(axis=1) / 10
test_resume['xgb_resume'] = xgb_test[['pre{}_{}_{}'.format('resume', 0, j) for j in range(0, 10)]].sum(axis=1) / 10

train_data = pd.read_feather('train_data.feather')
test_data = pd.read_feather('test_data.feather')


# In[13]:

print("STACKING")

lab_col = ['click','resume']
m_type='lgb'

valid_result = {}
pre_test = pd.DataFrame()
for label in lab_col:
    
    j=0
    
    if label == 'click':
        train_x = train_click
        train_y = train_data[train_data[label]!=-1][label].reset_index(drop=True)
        test = test_click
    else:
        train_x = train_resume
        train_y = train_data[train_data[label]!=-1][label].reset_index(drop=True)
        test = test_resume
    
    
    pre_valid = np.zeros((train_x.shape[0],1))
    
    #在处理后，每个样本已经独立，可以进行k折交叉验证
    seeds = [2021]
    for model_seed in range(len(seeds)):
        print("====================seeds{}====================".format(model_seed))
        kf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
        for train_index, valid_index in kf.split(train_x,train_y):
            print("************************"+m_type+":{}******************************".format(j+1))
            #v_train是本次训练的训练集
            #v_test是本次训练的测试集
            v_train_x,v_train_y =train_x.loc[train_index], train_y[train_index]
            v_valid_x,v_valid_y =train_x.loc[valid_index], train_y[valid_index]
            #通过关键字m_type调用相应模型
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 64,
                'max_depth':7,
                'learning_rate': 0.02,
                'feature_fraction': 0.85,
                'feature_fraction_seed':2021,
                'bagging_fraction': 0.85,
                'bagging_freq': 5,
                'bagging_seed':2021,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.5,
                'lambda_l2': 1.2,
                'verbose': -1
            }

            dtrain = lgb.Dataset(v_train_x, label=v_train_y)
            dvalid = lgb.Dataset(v_valid_x, label=v_valid_y)
            model = lgb.train(
                params,
                train_set = dtrain,
                num_boost_round=300,
                valid_sets = [dtrain, dvalid],
                verbose_eval=100,
            )

            if m_type=='cat':
                t = model.predict_proba(v_valid_x)[:,1]
                pre_valid[valid_index] = t.reshape(-1,1)
                scoret = model.predict_proba(test)[:,1]
                pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
            elif m_type=='xgb':
                t = model.predict(xgb.DMatrix(v_valid_x))
                pre_valid[valid_index] = t.reshape(-1,1)
                scoret = model.predict(xgb.DMatrix(test)) 
                pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
            elif m_type=='lgb':
                t = model.predict(v_valid_x)
                pre_valid[valid_index] = t.reshape(-1,1)
                scoret = model.predict(test) 
                pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
            else:
                t = model.predict_proba(v_valid_x)[:,1]
                pre_valid[valid_index] = t.reshape(-1,1)
                scoret = model.predict_proba(test)[:,1]
                pre_test['pre{}_{}_{}'.format(label, model_seed, j)] = scoret
                
            j = j+1
        
        valid_result[label] = pre_valid
            
        print(pre_valid[:5])
        print("AUC score: {}".format(roc_auc_score(train_y, pre_valid)))
        print("ACC score: {}".format(accuracy_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))
        print("F1 score: {}".format(f1_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))
        print("Precision score: {}".format(precision_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))
        print("Recall score: {}".format(recall_score(train_y, [1 if i >= 0.5 else 0 for i in pre_valid])))


# In[ ]:


pre_test['ctr'] = pre_test[['pre{}_{}_{}'.format('click', 0, j) for j in range(0, 5)]].sum(axis=1) / 5
pre_test['cvr'] = pre_test[['pre{}_{}_{}'.format('resume', 0, j) for j in range(0, 5)]].sum(axis=1) / 5


# In[ ]:


pre_test[['ctr', 'cvr']].to_csv('submit2.csv', header=False, index=False)


# In[ ]:


result1 = pd.read_csv('submit.csv', header=None)
result2 = pd.read_csv('submit.csv', header=None)


# In[ ]:


final_result = pd.DataFrame()
final_result['ctr'] = (result1['ctr'] + result2['ctr'] ) / 2
final_result['cvr'] = (result1['cvr'] + result2['cvr'] ) / 2


# In[ ]:


final_result.to_csv('final_submit.csv', header=False, index=False)

