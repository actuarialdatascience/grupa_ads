import numpy as np
import pandas as pd
from tensorflow.keras import Model, backend
from tensorflow.keras.layers import Dense, Input, Multiply


def read_data_set(dev_length=12):
    df_t = pd.read_csv('dane.csv', sep=';', index_col=False)

    # creating columns with cumulative amounts
    df_c = df_t[['Pay' + f"{x:02}" for x in range(dev_length)]].cumsum(axis=1)
    df_c.columns = ['PayCum' + f"{x:02}" for x in range(dev_length)]

    return pd.concat([df_t, df_c], axis=1)


df_full = read_data_set()
df = df_full[df_full["PayCum00"] > 0]

q = 20

dat_X = df[['LoB', 'cc', 'AY', 'AQ', 'age', 'inj_part']]
dat_C0 = df["PayCum00"]
dat_C1 = df["PayCum01"]
dat_Y = dat_C1 / np.sqrt(dat_C0)
dat_W = np.sqrt(dat_C0)

# creating neural network structure
features = Input(shape=dat_X.shape[1])
hidden_layer = Dense(units=q, activation='tanh')(features)
output_layer = Dense(units=1, activation=backend.exp)(hidden_layer)

volumes = Input(shape=1)
offset_layer = Dense(units=1, activation='linear', use_bias=False,
                     trainable=False)(volumes)
# originally there was also 'weights=[np.array(1).reshape(1, 1)]',
# but it works without it the same way
# and it seems that it does not add anything here

merged = Multiply()([output_layer, offset_layer])

model = Model(inputs=[features, volumes], outputs=merged)
model.compile(loss='mse', optimizer='rmsprop')
# fit neural network

model.fit([dat_X, dat_W], dat_Y,
          epochs=10,
          batch_size=1000,
          validation_split=0.1)
# predict claims dat$C1 and in - sample loss
model_predicted = model.predict([dat_X, dat_W]).flatten()
