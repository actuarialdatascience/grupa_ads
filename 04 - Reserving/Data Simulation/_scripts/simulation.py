import numpy as np


def neural_network_1(var1, inputs, m, textfilecontent):

    list_of_vars = textfilecontent['List.of.Variables']
    variable_layer_sizes = list_of_vars.loc[list_of_vars['Variable'] == var1]

    q1 = list_of_vars.loc[variable_layer_sizes].iloc[:, 1]
    q2 = list_of_vars.loc[variable_layer_sizes].iloc[:, 2]
    d1 = list_of_vars.loc[variable_layer_sizes].iloc[:, 3:].sum(axis=1)
    d2 = list_of_vars.loc[variable_layer_sizes].iloc[:, 3:]

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
