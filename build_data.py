"""Build the Train, Dev and Test data"""

import codecs
import json
import os
import random

import numpy as np


# Set random seeds for reproducibility
random.seed(2018)

# Define hyper parameters of the dataset
DATA_PARENT_DIR = "data"
# Sizes
TRAIN_SIZE = 8000
TEST_SIZE = 1000
DEV_SIZE = 1000
# Dimensions
TRAIN_DIMENSION_LOW = 10
TRAIN_DIMENSION_UPP = 200
TEST_DIMENSION_LOW = 1
TEST_DIMENSION_UPP = 400
# Scales
TRAIN_SCALE_LOW = 1
TRAIN_SCALE_UPP = 10
TEST_SCALE_LOW = 0.1
TEST_SCALE_UPP = 30


def get_one_example(dim_low, dim_up, scale_low, scale_up):
    """Build on example"""
    # TODO : (add random scale ?)
    dimension = random.randint(dim_low, dim_up)
    scale = random.uniform(scale_low, scale_up)
    x = [scale * random.uniform(-1., 1.) for _ in range(dimension)]
    y = np.linalg.norm(x, ord=1)
    return x, y


def export_to_file(data, file_path_x, file_path_y):
    """Write data to files, one example per line"""
    file_path_x = os.path.join(DATA_PARENT_DIR, file_path_x)
    file_path_y = os.path.join(DATA_PARENT_DIR, file_path_y)

    with codecs.open(file_path_x, 'w', encoding='utf-8') as f_x:
        with codecs.open(file_path_y, 'w', encoding='utf-8') as f_y:
            for x, y in data:
                f_x.write(" ".join(str(component) for component in x)  + "\n")
                f_y.write("{}\n".format(y))


def save_dict_to_json(d, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


if __name__ == "__main__":
    # Build the datasets
    data_train = [get_one_example(TRAIN_DIMENSION_LOW, TRAIN_DIMENSION_UPP,
                                  TRAIN_SCALE_LOW, TRAIN_SCALE_UPP)
                  for _ in range(TRAIN_SIZE)]
    data_dev = [get_one_example(TRAIN_DIMENSION_LOW, TRAIN_DIMENSION_UPP,
                                  TRAIN_SCALE_LOW, TRAIN_SCALE_UPP)
                  for _ in range(DEV_SIZE)]
    data_test = [get_one_example(TEST_DIMENSION_LOW, TEST_DIMENSION_UPP,
                                  TEST_SCALE_LOW, TEST_SCALE_UPP)
                  for _ in range(TEST_SIZE)]

    # Save the data to files
    if not os.path.exists(DATA_PARENT_DIR):
        os.makedirs(DATA_PARENT_DIR)

    export_to_file(data_train, "train.x", "train.y")
    export_to_file(data_dev, "dev.x", "dev.y")
    export_to_file(data_test, "test.x", "test.y")

    # Save datasets properties in json file
    sizes = {
        'train_size': len(data_train),
        'dev_size': len(data_dev),
        'test_size': len(data_test),
    }

    save_dict_to_json(sizes, os.path.join(DATA_PARENT_DIR, 'dataset_params.json'))
