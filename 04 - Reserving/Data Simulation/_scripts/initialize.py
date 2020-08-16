import os
import pandas as pd
import requests
import sys

from glob import iglob
from zipfile import ZipFile

URL = 'http://www.math.ethz.ch/~wueth/Simulation_Machines/Simulation.Machine.V1.zip'


def prepare_project(path: str):
    """
    Download the zipped file with project and unzip in the selected
    location.

    :param path: Path for storing unzipped project.
    :type path: str
    :return: None
    """
    zip_path = os.path.join(path, 'Simulation.Machine.V1.zip')

    # Download zip file with project
    requested_file = requests.get(URL)
    with open(zip_path, 'wb') as f:
        f.write(requested_file.content)

    # Extract contents
    with ZipFile(zip_path, 'r') as zip_obj:
        zip_obj.extractall(path)

    # Remove file
    os.remove(zip_path)


def add_path_dict(input_dict: dict, start_path: str, file_path: str):
    """
    Iteratively add contents of text files as pd.DataFrames to
    ``input_dict``.

    :param input_dict: Dictionary to be enriched by the data.
    :param start_path: Base path of the directory structure.
    :param file_path: Path to file to be read.
    :return: None
    """
    # Determine relative path
    relpath = os.path.relpath(file_path, start=start_path)

    # If only file remaining, store in dict, otherwise go 1 level deeper
    if relpath == os.path.basename(file_path):
        input_dict[os.path.splitext(relpath)[0]] = pd.read_csv(file_path,
                                                               sep='\t')
    else:
        parent_dir = relpath.split('/')[0]
        if parent_dir not in input_dict.keys():
            input_dict[parent_dir] = {}
        add_path_dict(input_dict=input_dict[parent_dir],
                      start_path=os.path.join(start_path, parent_dir),
                      file_path=file_path)


def read_project(path: str):
    """
    Iterate over .txt files in a directory and replicate their
    directory structure as a dictionary. Relies on the
    ``add_path_dict`` function.

    :param path: Base path of the directory structure to search.
    :return:
        Dictionary replicating the directory structure, with .txt
        files read.
    :rtype: dict
    """
    textfilecontent = {}

    # Discover .txt files and add them to the dictionary
    for filepath in iglob(os.path.join(path, '**/*.txt'), recursive=True):
        add_path_dict(input_dict=textfilecontent, start_path=path,
                      file_path=filepath)

    return textfilecontent


# Add path argument for unzipping
if __name__ == '__main__':
    """
    To download and unzip the project, run 
    ``python initialize.py <UNZIP_PATH>``.
    """
    prepare_project(sys.argv[1])
