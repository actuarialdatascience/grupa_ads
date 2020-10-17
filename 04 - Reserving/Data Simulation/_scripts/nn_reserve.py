import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from tensorflow.keras import Model, backend, initializers
from tensorflow.keras.layers import Dense, Input, Multiply


def fit_cl(tr, dev_length, hist_start):
    f = []
    for i in range(dev_length - 1):
        b = tr.loc[range(hist_start, hist_start + dev_length - i - 1),
                   f"PayCum{i + 1:02}"].sum()
        a = tr.loc[range(hist_start, hist_start + dev_length - i - 1),
                   f"PayCum{i:02}"].sum()
        f.append(b/a)
    return f


def fit_model_zero(df_cur, ay):
    # ay is the current accident year considered
    # narrowing down the data sets for relevant train data only
    df = df_cur[df_cur['AY'] < ay]
    remain = dev_length - (hist_end - ay)
    labels = [f"PayCum{(hist_end - ay + s):02}" for s in range(remain)]

    df_star = df[df[labels[0]] == 0]
    g = []
    for m in range(remain - 1):
        a = df_star[labels[m + 1]].sum()
        if m == 0:
            b = df[labels[0]].sum()
        else:
            b = df_star[labels[m]].sum()
        g.append(a / b)
        # narrowing down the data set, in line with formulas in the paper
        df_star = df_star[df_star['AY'] < ay - m - 1]

    return np.concatenate((np.zeros(hist_end - ay + 1), np.cumprod(g)))


def fit_model_nonzero(df_cur, d, q, epochs_p, b_p):
    # extracting only data points that are in training set
    # and relevant for neural network
    df = df_cur[df_cur[f"PayCum{d:02}"].notna() &
                df_cur[f"PayCum{d + 1:02}"].notna() &
                df_cur[f"PayCum{d:02}"] > 0]
    dat_x = df[['LoB', 'cc', 'AQ', 'age', 'inj_part']]
    dat_c0 = df[f"PayCum{d:02}"]
    dat_c1 = df[f"PayCum{d + 1:02}"]
    dat_y = dat_c1 / np.sqrt(dat_c0)
    dat_w = np.sqrt(dat_c0)

    features = Input(shape=dat_x.shape[1])
    hidden_layer = Dense(units=q, activation='tanh')(features)
    output_layer = Dense(units=1, activation=backend.exp)(hidden_layer)

    volumes = Input(shape=1)
    offset_layer = Dense(units=1, activation='linear',
                         use_bias=False, trainable=False,
                         kernel_initializer=initializers.Ones())(volumes)

    merged = Multiply()([output_layer, offset_layer])

    model = Model(inputs=[features, volumes], outputs=merged)
    model.compile(loss='mse', optimizer='rmsprop')
    # fit neural network

    model.fit([dat_x, dat_w], dat_y,
              epochs=epochs_p,
              batch_size=b_p,
              validation_split=0.1)

    return model


def read_data_set():
    df_t = pd.read_csv('dane.csv', sep=';', index_col=False)
    df_t.drop(['Unnamed: 0', 'ClNr', 'RepDel'], axis=1, inplace=True)
    df_t.drop([f"Open{j:02}" for j in range(dev_length)], axis=1, inplace=True)

    dev_periods = [f"{x:02}" for x in range(dev_length)]

    df_c = df_t[['Pay' + x for x in dev_periods]]
    df_c.columns = ['PayCum' + x for x in dev_periods]
    df_c = df_c.cumsum(axis=1)

    df_t = (pd.concat([df_t, df_c], axis=1)
            .groupby(['LoB', 'cc', 'AY', 'AQ', 'age', 'inj_part'])
            .sum()
            .reset_index())

    # removing strange data in the data sets - where payments are negative
    # to further consideration
    ew = (df_t[[f"Pay{j:02}" for j in range(dev_length)]].min(axis=1) >= 0)

    return df_t[ew]


dev_length = 12
df_0 = read_data_set()
hist_start = df_0['AY'].min()
hist_end = df_0['AY'].max()
types = ['Pay', 'PayCum']

df_train = df_0.copy()
for i in range(dev_length):
    df_train.loc[df_train['AY'] + i > hist_end,
                 [t + f"{i:02}" for t in types]] = np.nan

###############################

models = [fit_model_nonzero(df_cur=df_train,
                            d=x,
                            q=30,
                            epochs_p=100,
                            b_p=1000) for x in range(dev_length - 1)]
models_zero = [
    fit_model_zero(df_train, hist_start + x)
    for x in range(hist_end - hist_start + 1)
]

df_nonzero_predict = df_train.copy()

for j in range(dev_length - 1):
    ind_to_update = df_nonzero_predict[f"PayCum{j + 1:02}"].isna()
    df_temp = df_nonzero_predict[ind_to_update]
    dat_x = df_temp[['LoB', 'cc', 'AQ', 'age', 'inj_part']]
    dat_c0 = df_temp[f"PayCum{j:02}"]
#    print(dat_c0.describe())
    dat_w = np.sqrt(dat_c0)
    new_predicted = \
        models[j].predict([dat_x, dat_w]).flatten() * np.sqrt(dat_c0)
#    print((new_predicted - dat_c0).describe())
    df_nonzero_predict.loc[ind_to_update, f"PayCum{j + 1:02}"] = new_predicted


extract_columns = ['AY'] + [f"PayCum{j:02}" for j in range(dev_length)]
tr_0 = df_0[extract_columns].groupby('AY').sum()
tr_0_diag = np.diag(np.fliplr(tr_0))
models_zero_predict = tr_0_diag.reshape((-1, 1)) * models_zero

####

tr_non_zero_predict = df_nonzero_predict[extract_columns].groupby('AY').sum()
tr_predict = tr_non_zero_predict + models_zero_predict

####

model_cl = fit_cl(tr_0, dev_length, hist_start)

tr_cl_predict = tr_0.copy()
for i in range(dev_length - 1):
    curr_range = range(hist_end - i, hist_end + 1)
    basic = tr_cl_predict.loc[curr_range, f"PayCum{i:02}"]
    tr_cl_predict.loc[curr_range, f"PayCum{i + 1:02}"] = basic * model_cl[i]

####

tr_cl_predict.to_csv('tr_CL_predict.csv', sep=';', decimal=',')
tr_predict.to_csv('tr_predict.csv', sep=';', decimal=',')
tr_0.to_csv('tr_0.csv', sep=';', decimal=',')

print(tr_cl_predict[['PayCum11']].sum())
print(tr_predict[['PayCum11']].sum())
print(tr_0[['PayCum11']].sum())
