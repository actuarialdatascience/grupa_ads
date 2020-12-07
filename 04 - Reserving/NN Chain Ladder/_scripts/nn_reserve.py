import click
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from tensorflow.keras import Model, backend
from tensorflow.keras.initializers import Zeros, Ones, Constant
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
    diagonal = np.diag(np.fliplr(grouped))
    grouped.loc[:, :] = np.triu(grouped.values[:, ::-1])[:, ::-1]
    grouped.loc[:, "C_i,J"] = ay_totals
    grouped.loc[:, "Diagonal"] = diagonal

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
    for j in range(development_length):
        accumulated_factor = np.prod(f_list[j:])
        outstanding_claims.append(
            triangle.iloc[development_length - 1 - j, j] * accumulated_factor
        )

    return pd.Series(outstanding_claims[::-1], triangle.index)


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


def build_nn(q, initialize_cl, cl_df, dat_x=None):
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

    if not initialize_cl:
        output_layer = Dense(units=1, activation=backend.exp)(hidden_layer)
    else:
        output_layer = Dense(units=1, activation=backend.exp,
                             bias_initializer=Constant(value=cl_df),
                             kernel_initializer=Zeros()
                             )(hidden_layer)

    volumes = Input(shape=(1, ))
    offset_layer = Dense(units=1, activation='linear',
                         use_bias=False, trainable=False,
                         kernel_initializer=Ones())(volumes)

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


def extract_dat_x(df, per_batch_preproc):
    if per_batch_preproc:
        return df.loc[:, EXPLANATORY_COLUMNS].values
    else:
        return df.filter(
            regex="^({})".format("|".join(EXPLANATORY_COLUMNS))
        ).values


def fit_model_nonzero(df, dev_year, q, per_batch_preproc,
                      epochs, batch_size, initialize_cl):
    dat_x = extract_dat_x(df, per_batch_preproc)
    dat_c0 = df.loc[:, f"PayCum{dev_year:02}"].values
    dat_c1 = df.loc[:, f"PayCum{dev_year + 1:02}"].values
    dat_y = dat_c1 / np.sqrt(dat_c0)
    dat_w = np.sqrt(dat_c0)
    cl_df = np.log(dat_c1.sum() / dat_c0.sum())

    ################
    # Define network
    ################
    model = build_nn(q, initialize_cl, cl_df, dat_x)

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

    df_curr_ay = df[df['AY'] < ay]
    remain = development_length - (df.AY.max() - ay)
    labels = [f"PayCum{(df.AY.max() - ay + s):02}" for s in range(remain)]

    df_star = df_curr_ay[df_curr_ay[labels[0]] == 0]
    g = []
    for m in range(remain - 1):
        a = df_star[labels[m + 1]].sum()
        if m == 0:
            b = df[labels[0]].sum()
        else:
            b = df_star[labels[m]].sum()
        g.append(a / b if b > 0 else 1)
        # narrowing down the data set, in line with formulas in the paper
        df_star = df_star[df_star['AY'] < ay - m - 1]

    return np.prod(g if g != [] else 0)


def predict_zero_model(df, development_length, lob, model):
    extract_columns = ['AY'] \
                      + [f"PayCum{j:02}" for j in range(development_length)]
    tr = df[df.LoB == lob].loc[:, extract_columns].groupby('AY').sum()
    tr_diag = np.diag(np.fliplr(tr))
    current_zero = tr_diag * model
    return current_zero


def plot_by_variable(ret, variable, relative=False):
    plot_temp = (
        ret.loc[:, [variable, 'NN_Reserve', 'True_Reserve']]
            .groupby(variable)
            .agg(sum)
    )
    if relative:
        get_nn_vs_true = (
            lambda row:
            pd.Series({'NN_to_True': row['NN_Reserve'] / row['True_Reserve']})
        )
        plot_temp = pd.DataFrame(plot_temp.apply(get_nn_vs_true, axis=1))
    return plot_temp.plot.bar()


@click.command()
@click.option("--per-batch-preproc", is_flag=True)
@click.option("--initialize-cl", is_flag=True)
@click.argument("path")
def main(per_batch_preproc, initialize_cl, path):
    click.echo("Reading data...")
    df, development_length = read_data(path)
    ret = pd.DataFrame(df[EXPLANATORY_COLUMNS + ['AY']])
    ay_max = df.AY.max()

    click.echo("Computing per-LoB triangles...")
    lob_triangles = []
    for lob in range(1, df.LoB.max() + 1):
        current_triangle = compute_triangle(
            df[df.LoB == lob], development_length
        )
        current_cl_result = predict_cl(current_triangle, development_length)
        current_triangle.loc[:, 'CL'] = current_cl_result
        lob_triangles.append(current_triangle)

    click.echo("Training and predicting zero models...")
    for lob in range(1, df.LoB.max() + 1):
        models_zero_current_lob = []
        for accident_year in range(df.AY.min(), df.AY.max() + 1):
            current_model = fit_model_zero(df[df.LoB == lob], accident_year)
            models_zero_current_lob.append(current_model)
        current_zero_pred = predict_zero_model(df, development_length,
                                               lob, models_zero_current_lob)
        lob_triangles[lob - 1].loc[:, 'NN_zero'] = current_zero_pred
        print(lob_triangles[lob - 1])

    click.echo("Training nonzero models...")
    models = []
    if not per_batch_preproc:
        df = preprocess_data(df)
    for dev_year in range(development_length - 1):
        dy_train, dy_test = train_test_split(df, dev_year, development_length)
        current_model = fit_model_nonzero(
            dy_train, dev_year, 20, per_batch_preproc, 100, 10000,
            initialize_cl
        )
        current_model.save(f"model{dev_year}")
        models.append(current_model)

    click.echo("Predicting outstanding claims...")
    # Starting with fully available claim amount as of 0 - development year
    next_dev_year = df['PayCum00'].copy()
    for dev_year in range(development_length - 1):
        current_dev_year = next_dev_year
        indexes_to_update = (df.AY + dev_year >= df.AY.max())
        dat_x = extract_dat_x(df.loc[indexes_to_update, :], per_batch_preproc)
        dat_c0 = current_dev_year[indexes_to_update]
        dat_w = np.sqrt(dat_c0)
        pred = models[dev_year].predict([dat_x, dat_w]).flatten() * dat_w
        next_dev_year = df[f"PayCum{dev_year + 1:02}"].copy()
        next_dev_year[indexes_to_update] = pred

    # Preparing DataFrame with results of non-zero claims predictions
    ret['NN_Ult'] = pd.DataFrame(next_dev_year.rename('Ultimate'))
    ret['Diagonal'] = df.apply(lambda row:
                               row[f"PayCum{int(ay_max - row['AY']):02}"],
                               axis=1)
    ret['True_Ult'] = df[f"PayCum{development_length - 1:02}"]
    ret.drop(ret[ret.Diagonal == 0].index, inplace=True)
    ret['NN_Reserve'] = ret['NN_Ult'] - ret['Diagonal']
    ret['True_Reserve'] = ret['True_Ult'] - ret['Diagonal']
    ret.to_csv('Nonzero_results.csv', decimal=',', sep=';')

    # Initializing list with aggregate results per LoB
    aggregate_results = []

    click.echo("Combining results...")
    for lob in range(1, ret.LoB.max() + 1):
        ret_current_lob = ret.loc[ret.LoB == lob, :]
        tr_current_lob = lob_triangles[lob - 1]
        nonzero_pred = ret_current_lob.groupby('AY').agg(sum)['NN_Ult']

        tr_current_lob.loc[:, 'NN_nonzero'] = nonzero_pred
        tr_current_lob.loc[:, 'NN'] = (
            tr_current_lob.loc[:, ['NN_nonzero', 'NN_zero']].sum(axis=1)
        )

        tr_current_lob.loc[:, 'True_reserve'] = (
            tr_current_lob.loc[:, 'C_i,J'] - tr_current_lob.loc[:, 'Diagonal']
        )

        tr_current_lob.loc[:, 'NN_reserve'] = (
            tr_current_lob.loc[:, 'NN'] - tr_current_lob.loc[:, 'Diagonal']
        )

        tr_current_lob.loc[:, 'CL_reserve'] = (
            tr_current_lob.loc[:, 'CL'] - tr_current_lob.loc[:, 'Diagonal']
        )

        tr_current_lob.to_csv(f"triangle_lob{lob}.csv", decimal=',', sep=';')
        print(tr_current_lob)

        click.echo("Results for LoB " + str(lob) + ":")
        print(tr_current_lob[['C_i,J', 'CL', 'NN']].sum())
        aggregate_results.append(tr_current_lob[['C_i,J', 'CL', 'NN']].sum())

    click.echo("Total results:")
    print(pd.concat(aggregate_results, axis=1).agg(sum, axis=1))

    click.echo("Charts")
    variables = ['cc', 'age', 'LoB', 'AY', 'inj_part', 'cc']
    for var in variables:
        plot_by_variable(ret, var, relative=False)
        plot_by_variable(ret, var, relative=True)
    plt.show()

    click.echo("Finish")


if __name__ == "__main__":
    main()
