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
