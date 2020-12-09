import os

from pathlib import Path

from initialize import prepare_project, read_project
from feature_generation import feature_generation
from simulation import simulations


def main():
    repo_root_path = Path(__file__).parents[3]
    prepare_project(repo_root_path)
    texfilecontent = read_project(
        os.path.join(repo_root_path, "Simulation.Machine.V1")
    )

    features = feature_generation(
        500000,
        [0.25, 0.3, 0.2, 0.25],
        [0.01, 0.01, 0.01, 0.01],
        100,
        texfilecontent
    )

    outputs = simulations(features, texfilecontent)
    outputs.to_csv(os.path.join(os.path.dirname(__file__), "outputs.csv"))


if __name__ == "__main__":
    main()
