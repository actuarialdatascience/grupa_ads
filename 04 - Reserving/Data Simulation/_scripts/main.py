from initialize import prepare_project, read_project
from feature_generation import feature_generation
from simulation import simulations


def main():
    # prepare_project(".")
    texfilecontent = read_project(
        "/home/koscial/Repos/Simulation.Machine.V1"
    )

    # features = feature_generation(
    #     500000, [0.2, 0.2, 0.2, 0.2], [1, 1, 1, 1], 100, texfilecontent
    # )
    import pandas as pd
    features = pd.read_csv(
        "/home/koscial/Repos/Simulation.Machine.V1/features.csv"
    )

    outputs = simulations(features, texfilecontent)


if __name__ == "__main__":
    main()
