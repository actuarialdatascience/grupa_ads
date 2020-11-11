import numpy as np
import pandas as pd
import sys

from tensorflow.keras import Model, backend, initializers
from tensorflow.keras.layers import Dense, Input, Lambda, Multiply, concatenate
from tensorflow.keras.layers.experimental import preprocessing


def read_data(path):
    simulated_data = pd.read_csv(path, sep=';', index_col=False)

    # Determine the development length
    development_length = len(
        list(filter(lambda col: col.startswith("Pay"), simulated_data.columns))
    )

    # Drop unnecessary columns
    simulated_data.drop(
        ['ClNr'] + [f"Open{j:02}" for j in range(development_length)],
        axis=1,
        inplace=True
    )

    # Compute cumulative payments
    cumulative_payments = simulated_data[
        [f"Pay{j:02}" for j in range(development_length)]
    ]
    cumulative_payments.columns = [
        f"PayCum{j:02}" for j in range(development_length)
    ]
    cumulative_payments = cumulative_payments.cumsum(axis=1)

    # Concatenate
    simulated_data = pd.concat([simulated_data, cumulative_payments], axis=1)

    return simulated_data


def compute_triangle(df):
    # Determine the development length
    development_length = len(
        list(filter(lambda col: col.startswith("PayCum"), df.columns))
    )

    # Group and mask values 'in the future'
    grouped = (
        df[["AY"] + [f"PayCum{j:02}" for j in range(development_length)]]
        .groupby("AY")
        .agg(sum)
    )
    ay_totals = grouped.values[:, -1].copy()
    grouped.loc[:, :] = np.triu(grouped.values[:, ::-1])[:, ::-1]
    grouped.loc[:, "C_i,J"] = ay_totals

    return grouped


def predict_cl(triangle):
    """
    Compute predictions from the Chain Ladder method. This function assumes
    I = J.

    :param triangle: Claim triangle computed with ``compute_triangle``.
    :return:
    """
    # Determine accident years and development length
    development_length = len(
        list(filter(lambda col: col.startswith("PayCum"), triangle.columns))
    )

    # Compute development factors
    f_list = []
    for j in range(development_length - 1):
        c0 = triangle.iloc[:development_length - 1 - j, j].sum()
        c1 = triangle.iloc[:, j + 1].sum()
        f_list.append(c1/c0)

    # Compute predicted outstanding claims
    outstanding_claims = []
    for j in range(development_length - 1):
        accumulated_factor = np.prod(f_list[j:])
        outstanding_claims.append(
            triangle.iloc[development_length - 1 - j, j] * accumulated_factor
        )

    return pd.Series(outstanding_claims[::-1], triangle.index[1:])


def preprocess_data(df):
    # Determine accident years and development length
    development_length = len(
        list(filter(lambda col: col.startswith("PayCum"), df.columns))
    )

    # Filter data for claims observed prior to the last observed year
    df_filtered = (
        df
        .loc[df.AY + df.RepDel <= df.AY.max(), :]
        .drop(
            ["RepDel"] + [f"Pay{j:02}" for j in range(development_length)],
            axis=1
        )
    )

    # Apply one-hot-encoding
    df_preproc = pd.get_dummies(
        df_filtered,
        columns=["LoB", "cc", "inj_part"],
        sparse=True,
        drop_first=True
    )

    # Apply min-max scaling
    max_numerical = df_preproc[["AQ", "age"]].max(axis=0)
    min_numerical = df_preproc[["AQ", "age"]].min(axis=0)
    df_preproc[["AQ", "age"]] = (
        2 * (df_preproc[["AQ", "age"]] - min_numerical)
        / (max_numerical - min_numerical) - 1
    )

    return df_preproc


def build_nn(q, dat_x=None):
    if dat_x is not None and dat_x.size[1] == 5:
        inputs = Input(shape=(5, ), dtype="int32")

        encoders = []
        encoded = []
        for var_idx in range(5):
            if var_idx in [0, 1, 4]:
                current_encoder = preprocessing.CategoryEncoding(
                    output_mode="binary", sparse=True
                )
            else:
                current_encoder = preprocessing.Normalization()
            encoders.append(current_encoder)
            encoders[var_idx].adapt(dat_x[:, var_idx])
            encoded.append(encoders[var_idx](inputs[:, var_idx]))

        features = concatenate(encoded)
    elif dat_x is None or dat_x.size[1] == 146:
        features = Input(shape=(146, ))
    else:
        raise ValueError("The input array can have either 5 or 146 columns.")
    hidden_layer = Dense(units=q, activation='tanh')(features)
    output_layer = Dense(units=1, activation=backend.exp)(hidden_layer)

    volumes = Input(shape=(1, ))
    offset_layer = Dense(units=1, activation='linear',
                         use_bias=False, trainable=False,
                         kernel_initializer=initializers.Ones())(volumes)

    merged = Multiply()([output_layer, offset_layer])

    model = Model(inputs=[inputs, volumes], outputs=merged)
    model.compile(loss='mse', optimizer='rmsprop', metrics=["mse"])

    return model


def fit_model_nonzero(df, dev_year, q, per_batch_preproc, epochs, batch_size):
    # Determine accident years and development length
    development_length = len(
        list(filter(lambda col: col.startswith("PayCum"), df.columns))
    )

    # Prepare training data
    dat = (
        df.loc[
            (df[f"PayCum{dev_year:02}"] > 0)
            & (df.AY + df.RepDel <= df.AY.max()), :
        ]
        .drop(
            ["AY", "RepDel"]
            + [f"Pay{j:02}" for j in range(development_length)],
            axis=1
        )
    )

    if per_batch_preproc:
        dat_x = dat.iloc[:, :5].values
    else:
        dat_x = preprocess_data(dat).values
    dat_c0 = dat.loc[:, f"PayCum{dev_year:02}"].values
    dat_c1 = dat.loc[:, f"PayCum{dev_year + 1:02}"].values
    dat_y = dat_c1 / np.sqrt(dat_c0)
    dat_w = np.sqrt(dat_c0)

    ################
    # Define network
    ################
    model = build_nn(q, dat_x)

    # Fit network
    model.fit([dat_x, dat_w], dat_y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1)

    return model


def fit_model_zero(df, ay):
    """
    Fit the zero claims model by narrowing the training data to the relevant
    part only.

    :param df:
    :param ay: Current accident year
    :return:
    """
    # Determine accident years and development length
    development_length = len(
        list(filter(lambda col: col.startswith("PayCum"), df.columns))
    )
    hist_end = df.AY.max()

    df_curr_ay = df[df['AY'] < ay]
    remain = development_length - (hist_end - ay)
    labels = [f"PayCum{(hist_end - ay + s):02}" for s in range(remain)]

    df_star = df_curr_ay[df_curr_ay[labels[0]] == 0]
    g = []
    for m in range(remain - 1):
        a = df_star[labels[m + 1]].sum()
        if m == 0:
            b = df[labels[0]].sum()
        else:
            b = df_star[labels[m]].s
        g.append(a / b)
        # narrowing down the data set, in line with formulas in the paper
        df_star = df_star[df_star['AY'] < ay - m - 1]

    return np.concatenate((np.zeros(hist_end - ay + 1), np.cumprod(g)))


def main(path):
    print("Reading data...")
    df = read_data(path)

    print("Computing per-LoB triangles...")
    lob_triangles = []
    for lob in range(1, 5):
        current_triangle = compute_triangle(df[df.LoB == lob])
        current_triangle.to_csv(f"triangle_lob{lob}", index=False)
        lob_triangles.append(current_triangle)
        print(current_triangle)

    print("Training models...")
    models = []
    for dev_year in range(12):
        current_model = fit_model_nonzero(df, dev_year, 20, 100, 10000)
        current_model.save(f"model{dev_year}")
        models.append(current_model)

    print("Predicting outstanding claims...")

    print("Finish")


if __name__ == "__main__":
    main(sys.argv[1])
