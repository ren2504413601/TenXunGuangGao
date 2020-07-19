import pandas as pd
import numpy as np
import gc
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from collections import defaultdict


lgb_params_age = {
    'learning_rate' : 0.1,
    # 'min_child_samples': 5,
    'max_depth': 7,
    'lambda_l1': 2,
    # 'feature_fraction': .75,
    # 'bagging_fraction': .85,
    # 'seed': 99,
    'n_estimators': 3000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 10,
    'nthread': -1,
    'early_stopping_rounds': 100,
}

lgb_params_gender = {
    'n_estimators': 3000,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    # 'metric': 'None',
    # 'num_leaves': 63,
    # 'subsample': 0.8,
    # 'colsample_bytree': 0.8,
    'learning_rate': 0.05,
    'lambda_l2':2,
    'nthread': -1,
    # 'silent': True,
    'early_stopping_rounds': 100,
}

import sys
sys.path.append('/home/aistudio/external-libraries/lib/python3.7/site-packages')
import catboost as cbt

cat_params_age={
    "iterations":50000,
      "use_best_model":True,
    "random_seed":2020,
     "logging_level":'Verbose', 
    "early_stopping_rounds":200, 
    "loss_function":'MultiClass',
    'eval_metric':'MultiClass',
    'task_type':'GPU',
     'devices':'0'
}

cat_params_gender={
    "iterations":50000,
      "use_best_model":True,
    "random_seed":2020,
     "logging_level":'Verbose', 
    "early_stopping_rounds":200, 
    "loss_function":'Logloss',
    'eval_metric':'AUC',
    'task_type':'GPU',
     'devices':'0'
}

age_map = {i: i-1 for i in range(1,11)}
age_map_rev = {v:k for k,v in age_map.items()}
gender_map = {1: 1, 2: 0}
gender_map_rev = {v:k for k,v in gender_map.items()}

fold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)

def run_oof_lgb(X, y, X_test, lgb_params, kfold, label_size):
    models = []
    if label_size > 2:
        pred = np.zeros((X_test.shape[0], label_size))
        oof = np.zeros((X.shape[0], label_size))
    else:
        oof = np.zeros(X_train.shape[0])
        pred = np.zeros(X_test.shape[0]) 
    for index, (train_idx, val_idx) in enumerate(kfold.split(X, y)):

        train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])

        model = lgb.train(lgb_params, train_set, valid_sets=[train_set, val_set],verbose_eval=100)
        models.append(model)

        val_pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = val_pred
        val_y = y.iloc[val_idx]
        if label_size > 2:
            val_pred = np.argmax(val_pred, axis = 1)
        else:
            val_pred = np.round(val_pred)

        print(index+1, 'val acc:', metrics.accuracy_score(val_y, val_pred))
        test_pred = model.predict(X_test)
        pred += test_pred/kfold.n_splits
        del train_set, val_set, val_pred, val_y, test_pred
        gc.collect()
    return models, pred, oof

def run_oof_cat(X, y, X_test, cat_params, kfold, label_size):
    models = []
    pred = np.zeros((X_test.shape[0], label_size))
    oof = np.zeros((X.shape[0], label_size))
    for index, (trn_idx, val_idx) in enumerate(kfold.split(X=X, y=y)):
        print('-' * 88)
        x_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        trn_pool = cbt.Pool(x_trn, y_trn)
        val_pool = cbt.Pool(x_val, y_val)
        model = cbt.CatBoostClassifier(**cat_params)
        model.fit(trn_pool, eval_set=val_pool, verbose=500)
        models.append(model)
        val_pred = model.predict_proba(x_val)
        oof[val_idx] = val_pred

        val_pred = np.argmax(val_pred, axis=1)
        pred += (model.predict_proba(X_test) / kfold.n_splits)
        print(index+1, 'val acc:', metrics.accuracy_score(y_val, val_pred))
        del x_trn, y_trn, x_val, y_val, val_pred
        del trn_pool, val_pool
        gc.collect()
        
    return models, pred, oof

from sklearn.model_selection import train_test_split
def run_cat(X, y, X_test, cat_params):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.01, random_state = 2020)

    print('-' * 88)
    
    trn_pool = cbt.Pool(X_train, y_train)
    val_pool = cbt.Pool(X_val, y_val)
    model = cbt.CatBoostClassifier(**cat_params)
    model.fit(trn_pool, eval_set = val_pool, verbose = 500)

    pred = model.predict_proba(X_test)
   
    return pred

from keras.layers import *
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *

maxlen = 128
# max_features = 100000
embed_size = 300

def create_embedding(texts, emb_save_file):
    embedding_wv = KeyedVectors.load_word2vec_format(
            emb_save_file, binary=False)
    embedding_matrix = np.zeros((len(texts), embed_size))
    for ix, word in enumerate(texts):
        try:
            embedding_vector = embedding_wv[word]
            embedding_matrix[i] = embedding_vector
        except:
            continue
            
    return embedding_matrix


def model_embedding(out_size, embedding_matrix):
    max_features, embed_size = embedding_matrix.shape
    model = Sequential()
    model.add(Embedding(max_features, embed_size, 
                embeddings_initializer = K.constant(embedding_matrix),
                input_length = maxlen,
                trainable=False))
    
    model.add(Flatten())
    model.add(Dense(128))
    if out_size == 2:
        model.add(Dense(out_size, activation='sigmoid'))
        
    else:
        model.add(Dense(out_size, activation='softmax'))
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    # model.summary()
    return model

def model_lstm_atten(out_size, embedding_matrix):
    '''
    maxlen # max number of words in a question to use
    '''
    max_features, embed_size = embedding_matrix.shape
    
    inp = Input(shape = (maxlen,))
    
    x = Embedding(max_features, embed_size, 
                embeddings_initializer = K.constant(embedding_matrix),
                input_length = maxlen,
                trainable = False)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences = True))(x)
    y = Bidirectional(CuDNNGRU(64, return_sequences = True))(x)
    z = Conv1D(64, kernel_size = 1, kernel_initializer = initializers.he_uniform(seed=2020), activation = "tanh")(y)
    
    atten_1 = Attention(maxlen)(x) # skip connect
    atten_2 = Attention(maxlen)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    max_pool1 = GlobalMaxPooling1D()(z)
    
    convs = []
    filter_sizes = [2, 3, 5, 8]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=maxlen, kernel_size=fsz, activation='relu')(y)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    text_cnn = concatenate(convs, axis=1)
    
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool, max_pool1, text_cnn])
    conc = Dense(256)(conc)
    conc = BatchNormalization()(conc)
    conc = PReLU()(conc)
    conc = Dropout(0.2)(conc)
    
    conc = Dense(128)(conc)
    conc = BatchNormalization()(conc)
    if out_size == 2:
        outp = Dense(out_size, activation="sigmoid")(conc) 
    else:
        outp = Dense(out_size, activation="softmax")(conc)    

    model = Model(inputs = inp, outputs = outp)
    
    return model
    
cb = [EarlyStopping(monitor='val_loss', patience = 3, verbose = 1),]
               
def run_oof_nn(X, y, X_test, kfold, label_size):
    models = []
    pred = np.zeros((X_test.shape[0], label_size))
    oof = np.zeros((X.shape[0], label_size))
    for index, (trn_idx, val_idx) in enumerate(kfold.split(X=X, y=y)):
        K.clear_session()
        print('-' * 88)
        x_trn, y_trn = X[trn_idx], to_categorical(y.iloc[trn_idx], num_classes = label_size, dtype='int8')
        x_val, y_val = X[val_idx], to_categorical(y.iloc[val_idx], num_classes = label_size, dtype='int8')
        
        model = model_embedding(out_size = label_size)
        if label_size == 2:
            model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001), metrics=['acc'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.001), metrics=['acc']) 
        
        model.fit(x_trn, y_trn, batch_size=32, epochs=10, validation_data=(x_val, y_val), callbacks = cb, verbose=1)
        models.append(model)
        val_pred = model.predict(x_val, batch_size=32, verbose=1)
        oof[val_idx] = val_pred
        test_pred = model.predict(X_test, batch_size=32, verbose=1)
        pred += test_pred / kfold.n_splits
        print(index+1, 'val acc:', metrics.accuracy_score(y_val, np.argmax(val_pred, axis=1)))
        del x_trn, y_trn, x_val, y_val, val_pred, test_pred
        del model
        gc.collect()
        
    return models, pred, oof

def run_nn(X, y, X_test, label_size, embedding_matrix):
    K.clear_session()
    print('-' * 88)
    x_trn, y_trn = X, to_categorical(y, num_classes = label_size, dtype='int8')
    model = model_embedding(label_size, embedding_matrix)
    if label_size == 2:
        model.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001), metrics=['acc'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.001), metrics=['acc'])
    
    model.fit(x_trn, y_trn, batch_size=32, epochs=1, validation_split=0.1, shuffle = True, callbacks = cb, verbose=1)
    pred = model.predict(X_test, batch_size=32, verbose=1)
    
    return pred


def get_feat_importance(models):
    ret = []
    for index, model in enumerate(models):
        df = pd.DataFrame()
        '''
        # lgb
        df['name'] = model.feature_name()
        df['score'] = model.feature_importance()
        df['fold'] = index
        '''
        # catboost
        df['name'] = model.feature_names_
        df['score'] = model.feature_importances_
        df['fold'] = index
        ret.append(df)
    df = pd.concat(ret)
    df = df.groupby('name', as_index=False)['score'].mean()
    df.sort_values(['score'], ascending=False)
    return df

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
    
def group_df(df, group_id, feat):
    df[feat] = df[feat].astype(str)
    return df.groupby(group_id)[feat].agg(list)


if __name__ == "__main__":
    # Load data
    train_path = "work/train_preliminary/"
    train_user = pd.read_csv(train_path + "user.csv")
    Feats = pd.read_csv('features/300_w2v_creative_id_feats.csv', dtype = np.float32)
    # del Feats['Unnamed: 0']
    # del t['Unnamed: 0']
    # gc.collect()
    for df_file in [
        'features/300_w2v_advertiser_id_feats.csv','features/group_feats.csv', 
        'features/ad_id_w2v_feats.csv',
         'features/creative_id_w2v_feats.csv', 'features/advertiser_id_w2v_feats.csv',
          'features/advertiser_id_tfidf_feats.csv', 'features/creative_id_tfidf_feats.csv',
        'features/ad_id_tfidf_feats.csv', 'features/300_w2v_ad_id_feats.csv'
        ]:
    # for df_file in ['features/300_w2v_advertiser_id_feats.csv','features/group_feats.csv', 'features/300_w2v_ad_id_feats.csv']:
        Feats = Feats.merge(pd.read_csv(df_file, dtype = np.float32), on = 'user_id', how = 'left')
    Feats = Feats.set_index('user_id')
    X_train = Feats.loc[train_user.user_id].sort_index()
    X_test = Feats.drop(train_user.user_id, axis = 0).sort_index()
    # del Feats
    # gc.collect()

    # if os.path.isfile('work/df_data.csv'):
    #     df_data = pd.read_csv('work/df_data.csv')
    # else:
    #     df_data = genrate_df_data()

    # data_frame = group_df(df_data, 'user_id', 'creative_id')
    

    # trainIdx = train_user.user_id
    # testIdx = [ix for ix in data_frame.index if ix not in trainIdx]
    # X_train = pad_sequences(np.array(data_frame.loc[trainIdx].sort_index().apply(np.array)), maxlen=maxlen)
    # X_test = pad_sequences(np.array(data_frame.loc[testIdx].sort_index().apply(np.array)), maxlen=maxlen)
    
    print(X_train.shape, X_test.shape)
    y_age = train_user[['age','user_id']].set_index("user_id").sort_index()
    y_gender = train_user[['gender','user_id']].set_index("user_id").sort_index()

    y_age.age = y_age.age.map(age_map)
    y_gender.gender = y_gender.gender.map(gender_map)
    

    # oof train
    # catboost train
    # age_models, age_pred, age_oof = run_oof_cat(X_train, y_age, X_test, cat_params_age, fold, 10)
    # gender_models, gender_pred, gender_oof = run_oof_cat(X_train, y_gender, X_test, cat_params_gender, fold, 2)

    age_pred = run_cat(X_train, y_age, X_test, cat_params_age)
    gender_pred = run_cat(X_train, y_gender, X_test, cat_params_gender)
    # nn train
    # age_models, age_pred, age_oof = run_oof_nn(X_train, y_age, X_test, fold, label_size = 10)
    # gender_models, gender_pred, gender_oof = run_oof_nn(X_train, y_gender, X_test, fold, label_size = 2)
    # min_times = 3
    # dict = defaultdict(int)
    # texts = []
    # for token in data_frame.values:
    #     for t in token:
    #         dict[t] += 1
    #         if dict[t] >= min_times:
    #             texts.append(t)

    # embedding_matrix = create_embedding(texts, emb_save_file = 'features/w2v_300.txt')
    
    # age_pred = run_nn(X_train, y_age, X_test, 10, embedding_matrix)
    # gender_pred = run_nn(X_train, y_gender, X_test, 2, embedding_matrix)
    # submit
    sub = pd.DataFrame()

    sub['user_id'] = X_test.index
    sub['predicted_age'] = np.argmax(age_pred, axis = 1).astype(int)
    sub['predicted_gender'] = np.argmax(gender_pred, axis = 1).astype(int)
    sub["predicted_gender"] = sub["predicted_gender"].map(gender_map_rev)
    sub["predicted_age"] = sub["predicted_age"].map(age_map_rev)
    sub.to_csv("./submission.csv", header = True, index = False, encoding='utf-8')