import numpy as np
import pandas as pd


def neural_network_1(var1, inputs, m, textfilecontent):

    list_of_vars = textfilecontent['List.of.Variables']
    variable_layers = list_of_vars.loc[list_of_vars['Variable'] == var1]

    q1 = variable_layers.iloc[:, 1]
    q2 = variable_layers.iloc[:, 2]
    d1 = variable_layers.iloc[:, 3:].sum(axis=1)
    d2 = variable_layers.iloc[:, 3:]

    data2 = np.ones((m, 1))

    for i in range(len(d2.columns)):
        if d2.iloc[0, i] == 1 and i in [0, 1, 5]:
            translator = textfilecontent['Translators'][var1][d2.columns[i]]
            data2 = np.hstack((data2, translator.loc[inputs[:, i]]))
        elif d2.iloc[0, i] == 1:
            translator = textfilecontent['Translators'][var1][d2.columns[i]]
            temp = np.round(
                2 * np.minimum(
                    np.maximum(
                        inputs[:, i+1],
                        np.array(translator.iloc[0, :])
                    ),
                    np.array(translator.iloc[1, :])
                ) / int(translator.iloc[1, :] - translator.iloc[0, :]) - 1,
                2).reshape(-1, 1)
            data2 = np.hstack((data2, temp))

    beta = textfilecontent['Parameters'][var1]['beta']
    W1 = textfilecontent['Parameters'][var1]['W1']
    W2 = textfilecontent['Parameters'][var1]['W2']

    z1_j = np.ones((int(q1)+1, m))
    z1_j[1:, :] = np.power(1 + np.exp(- W1.dot(np.transpose(data2))), -1)
    z2_j = np.ones((int(q2) + 1, m))
    z2_j[1:, :] = np.power(1 + np.exp(- W2.dot(2 * z1_j - 1)), -1)
    mu_t = np.transpose(beta).dot(2 * z2_j - 1)
    return mu_t


def neural_network_2(var1, inputs, m, textfilecontent):

    list_of_vars = textfilecontent['List.of.Variables']
    variable_layers = list_of_vars.loc[list_of_vars['Variable'] == var1]

    q1 = variable_layers.iloc[:, 1]
    q2 = variable_layers.iloc[:, 2]
    d1 = variable_layers.iloc[:, 3:].sum(axis=1)
    d2 = variable_layers.iloc[:, 3:]

    data2 = np.ones((m, 1))

    for i in range(len(d2.columns)):
        if d2.iloc[0, i] == 1:
            translator = textfilecontent['Translators'][var1][d2.columns[i]]
            temp = np.round(
                2 * np.minimum(
                    np.maximum(
                        inputs[:, i+1],
                        np.array(translator.iloc[0, :])
                    ),
                    np.array(translator.iloc[1, :])
                ) / int(translator.iloc[1, :] - translator.iloc[0, :]) - 1,
                2).reshape(-1, 1)
            data2 = np.hstack((data2, temp))

    beta = textfilecontent['Parameters'][var1]['beta']
    W1 = textfilecontent['Parameters'][var1]['W1']
    W2 = textfilecontent['Parameters'][var1]['W2']

    z1_j = np.ones((int(q1)+1, m))
    z1_j[1:, :] = np.power(1 + np.exp(- W1.dot(np.transpose(data2))), -1)
    z2_j = np.ones((int(q2) + 1, m))
    z2_j[1:, :] = np.power(1 + np.exp(- W2.dot(2 * z1_j - 1)), -1)
    mu_t = np.exp(np.transpose(beta).dot(2 * z2_j - 1))
    pi_t = mu_t / mu_t.sum(axis=0)
    return pi_t


def neural_network_3(var1, inputs, m, textfilecontent):

    list_of_vars = textfilecontent['List.of.Variables']
    variable_layers = list_of_vars.loc[list_of_vars['Variable'] == var1]

    q = variable_layers.iloc[:, 1]
    d1 = variable_layers.iloc[:, 3:].sum(axis=1)
    d2 = variable_layers.iloc[:, 3:]

    data2 = np.ones((m, 1))

    for i in range(13):
        if d2.iloc[0, i] == 1:
            translator = textfilecontent['Translators'][var1][d2.columns[i]]
            temp = np.round(
                2 * np.minimum(
                    np.maximum(
                        inputs[:, i+1],
                        np.array(translator.iloc[0, :])
                    ),
                    np.array(translator.iloc[1, :])
                ) / int(translator.iloc[1, :] - translator.iloc[0, :]) - 1,
                2).reshape(-1, 1)
            data2 = np.hstack((data2, temp))

    data2 = np.hstack((data2, inputs[['Pay00', 'Pay01', 'Pay02', 'Pay03', 'Pay04', 'Pay05',
                                      'Pay06', 'Pay07', 'Pay08', 'Pay09', 'Pay10', 'Pay11']]))

    beta = textfilecontent['Parameters'][var1]['beta']
    W = textfilecontent['Parameters'][var1]['W']

    z_j = np.ones((int(q)+1, m))
    z_j[1:, :] = np.power(1 + np.exp(- W.dot(np.transpose(data2))), -1)
    pi0 = np.power(1 + np.exp(np.transpose(beta).dot(2 * z_j - 1)), -1)
    return pi0


def cash_flow_prep_1(nop, input2, art_obs, seed1, textfilecontent):

    # We only look at observations with exactly np payments
    x_np = input2[input2['K'] == nop]

    # We add an artificial observation
    x_np.loc[len(x_np.index)] = art_obs

    # We simulate the proportions paid in the positive payments
    # Get the output of the neural network
    x_np_nr = x_np.shape[0]
    var1 = 'P' + nop + 'K'
    pi_t = neural_network_2(var1, x_np, x_np_nr, textfilecontent).reshape((-1, x_np_nr), order='F')

    # The possible distribution patterns are coded as follows
    distribution_codes = textfilecontent['Parameters'][var1]['distribution.codes']

    # It depends on the reporting delay what distribution patterns of the payments are possible:
    ldc = distribution_codes.shape[0]

    pi_t_helper = pi_t - pi_t * np.array(
        (np.repeat(distribution_codes, repeats=x_np_nr) % 2 ** (x_np[:, 7] + 1)) > 0
    ).reshape((ldc, -1))

    pi_t = pi_t_helper / np.sum(pi_t_helper, axis=0)
    # We have to separate the cases where none of the distribution.codes is possible because of the reporting delay

    # First we take the observations where we have at least one distribution pattern that is possible
    # We add one artificial probability vector
    indexing_helper = np.isnan(pi_t[0, :])

    pi_t_tilde = np.hstack((pi_t[:, indexing_helper], pi_t[:, -1])).reshape((ldc, -1), order='F')

    # Generate the distribution pattern according to pi_t.tilde
    np.random.seed(seed1 + 20 + nop)

    random_generation = np.random.uniform(size=pi_t_tilde.shape[1])
    cumul_pi_t = np.cumsum(pi_t_tilde / pi_t_tilde.sum(axis=0), axis=0)

    distribution_pattern = distribution_codes[np.array(random_generation > cumul_pi_t.transpose()).sum(axis=0)]

    # Add the distribution pattern (correcting for the artificially added probability vector)

    x_np.loc[indexing_helper, 26] = distribution_pattern[, :-1]

    # For the observations with reporting delay equal to 1, only few patterns are available
    # Thus, for a random half of these observations, we distribute the payments arbitrarily
    # First we add an artificial observation to prevent the code from crashing
    art_obs_new = art_obs
    art_obs_new['V27'] = None
    art_obs_new['RepDel'] = -1
    x_np.loc[len(x_np.index)] = art_obs_new
    pi_t = np.hstack((pi_t, pi_t[:, [-1]]))
    pi_t[:, -1] = np.nan

    np.random.seed(seed1 + 100 + nop)

    temp = np.where(x_np['RepDel'] == 1)[0]
    choose = np.random.choice(range(len(temp)), size=np.ceil(len(temp)/2))
    np.random.seed(seed1 + 120 + nop)
    samp = np.random.choice(range(2, 13), size=choose.shape[0] * nop).reshape((nop, -1))
    x_np[temp[choose], -1] = np.sum(2 ^ samp, axis=1)

    # Now we take the observations where we have no distribution pattern that is possible
    # (because of the reporting delay)

    np.random.seed(seed1 + 140 + nop)
    samp = np.vstack([np.random.choice(range(x + 1, 13), nop) for x in x_np.iloc[indexing_helper, 7]])
    x_np.iloc[indexing_helper, 26] = np.sum(2 ^ samp, axis=1)

    # Correct for the observations that we added artificially
    x_np = x_np[x_np['ClNr'] != -1]

    # Output of the function
    return x_np


def cash_flow_prep_2(nop, input2, art_obs, textfilecontent):

    # We only look at observations with no recovery payment
    # We add an artificial observation
    x_np_0 = input2[input2['Zmnew'] == 0].append(art_obs)

    # We simulate the proportions paid in the positive payments
    # Get the output of the neural network
    pi_t = neural_network_2('P' + nop + 'pRA', x_np_0, x_np_0.shape[0], textfilecontent)
    pi_t = pi_t.reshape((-1, x_np_0.shape[0]), order='F')

    # Determine the cash flow
    x_np_0.loc[:, 15:26] = cash_flow_pattern_1(x_np_0, pi_t)

    # Output of the function
    return x_np_0
