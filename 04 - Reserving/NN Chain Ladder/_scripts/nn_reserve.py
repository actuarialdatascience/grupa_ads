import click
import numpy as np
import pandas as pd
import sys

from tensorflow.keras import Model, backend, initializers
from tensorflow.keras.layers import Dense, Input, Lambda, Multiply, concatenate
from tensorflow.keras.layers.experimental import preprocessing

EXPLANATORY_COLUMNS = ["AQ", "age", "LoB", "cc", "inj_part"]


def read_data(path):
    simulated_data = pd.read_csv(path, sep=';', index_col=False)

    # Determine the development length
    development_length = len(
        list(filter(lambda col: col.startswith("Pay"), simulated_data.columns))
    )

    # Drop unnecessary columns
    simulated_data.drop(
        ['ClNr']
        + ['RepDel']
        + [f"Open{j:02}" for j in range(development_length)],
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
    simulated_data = (
        pd.concat([simulated_data, cumulative_payments], axis=1)
            .groupby(['LoB', 'cc', 'AY', 'AQ', 'age', 'inj_part'])
            .agg(sum)
            .reset_index()
    )

    return simulated_data, development_length


def compute_triangle(df, development_length):
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


def predict_cl(triangle, development_length):
    """
    Compute predictions from the Chain Ladder method. This function assumes
    I = J.

    :param triangle: Claim triangle computed with ``compute_triangle``.
    :param development_length: Maximum development periods.
    :return:
    """
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
    other_columns = set(df.columns).difference(EXPLANATORY_COLUMNS)
    other_data = df[other_columns]
    explanatory_data = df[EXPLANATORY_COLUMNS]

    # Apply one-hot-encoding
    preproc_data = pd.get_dummies(
        explanatory_data,
        columns=["LoB", "cc", "inj_part"],
        sparse=True,
        drop_first=True
    )

    # Apply min-max scaling
    max_numerical = preproc_data[["AQ", "age"]].max(axis=0)
    min_numerical = preproc_data[["AQ", "age"]].min(axis=0)
    preproc_data[["AQ", "age"]] = (
        2 * (preproc_data[["AQ", "age"]] - min_numerical)
        / (max_numerical - min_numerical) - 1
    )

    return pd.concat([preproc_data, other_data], axis=1)


def build_nn(q, dat_x=None):
    if dat_x is not None and dat_x.shape[1] == 5:
        features = Input(shape=(5, ), dtype="int32")

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
            encoded.append(encoders[var_idx](features[:, var_idx]))

        features_encoded = concatenate(encoded)
        hidden_layer = Dense(units=q, activation='tanh')(features_encoded)
    elif dat_x is None or dat_x.shape[1] > 5:
        features = Input(shape=(dat_x.shape[1], ))
        hidden_layer = Dense(units=q, activation='tanh')(features)

    output_layer = Dense(units=1, activation=backend.exp)(hidden_layer)

    volumes = Input(shape=(1, ))
    offset_layer = Dense(units=1, activation='linear',
                         use_bias=False, trainable=False,
                         kernel_initializer=initializers.Ones())(volumes)

    merged = Multiply()([output_layer, offset_layer])

    model = Model(inputs=[features, volumes], outputs=merged)
    model.compile(loss='mse', optimizer='rmsprop', metrics=["mse"])

    return model


def train_test_split(df, dev_year, development_length):
    # Compute logical pd.Series for train/test filtering
    logical_train = (
        (df[f"PayCum{dev_year:02}"] > 0)
        & (df.AY + dev_year < df.AY.max())
    )

    # Split and drop redundant column
    cols_to_drop = ["AY"] + [f"Pay{j:02}" for j in range(development_length)]
    df_train = df.loc[logical_train, :].drop(cols_to_drop, axis=1)
    df_test = df.loc[~logical_train, :].drop(cols_to_drop, axis=1)

    return df_train, df_test


def fit_model_nonzero(df, dev_year, q, per_batch_preproc, epochs, batch_size):
    if per_batch_preproc:
        dat_x = df.loc[:, EXPLANATORY_COLUMNS].values
    else:
        dat_x = df.filter(
            regex="^({})".format("|".join(EXPLANATORY_COLUMNS))
        ).values
    dat_c0 = df.loc[:, f"PayCum{dev_year:02}"].values
    dat_c1 = df.loc[:, f"PayCum{dev_year + 1:02}"].values
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


@click.command()
@click.option("--per-batch-preproc", is_flag=True)
@click.argument("path")
def main(per_batch_preproc, path):
    click.echo("Reading data...")
    df, development_length = read_data(path)

    click.echo("Computing per-LoB triangles...")
    lob_triangles = []
    for lob in range(1, df.LoB.max() + 1):
        current_triangle = compute_triangle(
            df[df.LoB == lob], development_length
        )
        current_triangle.to_csv(f"triangle_lob{lob}", index=False)
        lob_triangles.append(current_triangle)
        print(current_triangle)

    click.echo("Training models...")
    models = []
    if not per_batch_preproc:
        df = preprocess_data(df)
    for dev_year in range(development_length - 1):
        dy_train, dy_test = train_test_split(df, dev_year, development_length)
        current_model = fit_model_nonzero(
            dy_train, dev_year, 20, per_batch_preproc, 100, 10000
        )
        current_model.save(f"model{dev_year}")
        models.append(current_model)

    click.echo("Predicting outstanding claims...")

    click.echo("Finish")


if __name__ == "__main__":
    main()
