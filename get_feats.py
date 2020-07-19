import pandas as pd
import numpy as np
import gc
import multiprocessing as mp
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

def genrate_df_data():
    # Load data
    train_path = "work/train_preliminary/"
    test_path = "work/test/"
    train_ad = pd.read_csv(train_path + "ad.csv")
    train_click_log = pd.read_csv(train_path + "click_log.csv")
    test_ad = pd.read_csv(test_path + "ad.csv")
    test_click_log = pd.read_csv(test_path + "click_log.csv")
    df_train = train_ad.merge(train_click_log, on = "creative_id", how = "left")
    df_test = test_ad.merge(test_click_log, on = "creative_id", how = "left")
    df_data = pd.concat([df_train, df_test], ignore_index = True)
    df_data = df_data.replace('\\N', -1)
    # save
    df_data.to_csv('work/df_data.csv', index = False)
    del train_ad, train_click_log, test_ad, test_click_log, df_train, df_test
    gc.collect()
    return df_data

# https://github.com/MichaelYin1994/tianchi-trajectory-data-mining/blob/master/source_code/embedding_signal_sequence.py
def hashfxn(astring):
    return ord(astring[0])

def group_df(df, group_id, feat):
    df[feat] = df[feat].astype(str)
    return df.groupby(group_id)[feat].agg(list).reset_index()
     
def trian_save_word2vec(df, group_id, feat, length, saved_file ='w2v.txt'):
    data_frame = group_df(df, group_id, feat)
    model = Word2Vec(data_frame[feat].values,
                     size=length,
                     window= 32,
                     min_count = 1,
                     workers = mp.cpu_count(),
                     sg = 1,
                     iter = 10,
                     seed = 1,
                     )
    model.wv.save_word2vec_format(saved_file)
    print("w2v model done")
    return model.wv, data_frame

def get_w2v_feat(df, group_id, feat, length, saved_file):
    if os.path.isfile(saved_file):
        embedding_wv = KeyedVectors.load_word2vec_format(
            saved_file, binary=False)
        data_frame = group_df(df, group_id, feat)
    else:
        embedding_wv, data_frame = trian_save_word2vec(df, group_id, feat, length, saved_file)
        
    data_frame[feat] = data_frame[feat].apply(
        lambda x: np.array([[embedding_wv[c] if c in embedding_wv.vocab.keys() else [0]*length] for c in x])
    )
    
    for m in range(length):
        data_frame['w2v_{}_mean'.format(m)] = data_frame[feat].apply(
            lambda x: x[:, m].mean()
        )
    del data_frame[feat], embedding_wv
    gc.collect()
    return data_frame

# https://github.com/MichaelYin1994/tianchi-trajectory-data-mining/blob/master/source_code/embedding_geo_info.py
def w2v_feat(df, group_id, feat, length, saved_file):
    if os.path.isfile(saved_file):
        embedding_wv = KeyedVectors.load_word2vec_format(
            saved_file, binary=False)
        data_frame = group_df(df, group_id, feat)
    else:
        embedding_wv, data_frame = trian_save_word2vec(df, group_id, feat, length, saved_file)

    # Sentance vector
    embedding_vec = []
    for seq in data_frame[feat].values:
        seq_vec, word_count = 0, 0
        for word in seq:
            if word not in embedding_wv.vocab.keys():
                continue
            else:
                seq_vec += embedding_wv[word]
                word_count += 1
        if word_count == 0:
            embedding_vec.append(length * [0])
        else:
            embedding_vec.append(seq_vec / word_count)

    embedding_df = pd.DataFrame(np.array(embedding_vec), 
        columns=["embedding_{}_{}".format(feat, i) for i in range(length)])
    embedding_df[group_id] = data_frame[group_id]
        
    del data_frame, embedding_wv, embedding_vec
    gc.collect()
    return embedding_df

# 构造统计特征
def group_feature(df, key, target, aggs):   
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    df[target] = df[target].astype(float)
    t = df.groupby(key)[target].agg(agg_dict).reset_index()
    return t

# counter, tfidf特征
def count_tfidf(df, group_id, group_target, num):
    df[group_target] = df[group_target].astype(str)
    tmp = df.groupby(group_id)[group_target].agg(list).reset_index()
    tmp[group_target] = tmp[group_target].apply(lambda x: ' '.join(x))

    tfidf_enc_tmp = TfidfVectorizer(max_features = 100000, min_df = 3)
    tfidf_vec_tmp = tfidf_enc_tmp.fit_transform(tmp[group_target])
    svd_tag_tmp = TruncatedSVD(n_components=num, n_iter=5, random_state=52)
    tag_svd_tmp = svd_tag_tmp.fit_transform(tfidf_vec_tmp)
    tag_svd_tmp = pd.DataFrame(tag_svd_tmp)
    tag_svd_tmp.columns = ['{}_tfidf_{}'.format(group_target, i)
                           for i in range(num)]

    countvec = CountVectorizer(max_features = 100000, min_df = 3)
    count_vec_tmp = countvec.fit_transform(tmp[group_target])
    svd_tmp = TruncatedSVD(n_components=num, n_iter=5, random_state=52)
    svd_tmp = svd_tmp.fit_transform(count_vec_tmp)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(group_target, i)
                       for i in range(num)]

    return pd.concat([tmp[[group_id]], tag_svd_tmp, svd_tmp], axis=1)

if __name__ == "__main__":

    # Load data
    # train_path = "work/train_preliminary/"
    # test_path = "work/test/"
    # train_ad = pd.read_csv(train_path + "ad.csv")
    # train_click_log = pd.read_csv(train_path + "click_log.csv")
    # test_ad = pd.read_csv(test_path + "ad.csv")
    # test_click_log = pd.read_csv(test_path + "click_log.csv")
    # df_train = train_ad.merge(train_click_log, on = "creative_id", how = "left")
    # df_test = test_ad.merge(test_click_log, on = "creative_id", how = "left")
    # df_data = pd.concat([df_train, df_test], ignore_index = True)
    if os.path.isfile('work/df_data.csv'):
        df_data = pd.read_csv('work/df_data.csv')
    else:
        df_data = genrate_df_data()
    

    # Feats = df_data.drop_duplicates(subset = "user_id")[["user_id"]]

    # del train_ad, train_click_log, test_ad, test_click_log, df_train, df_test
    # gc.collect()

    # Get w2c feats
    feats_w2v = w2v_feat(df_data, "user_id", "advertiser_id", 300, "features/ad_id_w2v_300.txt")
    # Feats = Feats.merge(feats_w2v, on = "user_id", how = "left")
    feats_w2v.to_csv('./features/300_w2v_ad_id_feats.csv')

    # # Get group features
    # for col in ["creative_id", "ad_id", "product_category", "advertiser_id", "time", "click_times"]:
    #     t = group_feature(df_data, "user_id", col, ['max','min','mean','sum','std', 'nunique', 'count'])
    #     Feats = Feats.merge(t, on = "user_id", how = "left")


    # # Get counter ftidf fetures
    # for col in ["creative_id", "ad_id", "advertiser_id"]:
    #     t = get_count_tfidf(df_data, "user_id", col, 30)
    #     Feats = Feats.merge(t, on = "user_id", how = "left")

    # Feats.to_csv('./features/group_feats.csv')





