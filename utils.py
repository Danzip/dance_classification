import os
import re

import pandas as pd
from movinets import MoViNet as mn
from movinets.config import _C

KINETICS_CLASSES_CSV = "kinetics_600_classes.csv"

SEPERATE_CLASSES = "separate_classes"
KINETICS_PATH = os.path.join(SEPERATE_CLASSES, KINETICS_CLASSES_CSV)
DANCES_PATH_5 = os.path.join(SEPERATE_CLASSES, 'dances_5.csv')
DANCES_PATH_18 = os.path.join(SEPERATE_CLASSES, 'dances_18.csv')
KINETICS_600_CSV = os.path.join(SEPERATE_CLASSES, "kineticks600_classes.csv")
CLASS_NUM = "class_num"
CLASS_LABEL = "class_label"
NUM_FRAMES = 8
model_id = "a0"
RESOLUTION = 224
OUTPUT_SIZE = (RESOLUTION, RESOLUTION)
BATCH_SIZE = 8


def extract_number(string):
    pattern = r'\d+'  # Matches one or more digits
    match = re.search(pattern, string)

    if match:
        return int(match.group())
    else:
        return None


def load_movinet_model(model_id):
    model_name = f"MoViNet{model_id.upper()}"
    num = extract_number(model_name)
    casual = True if num < 3 else False  # package movinets doesn't support streaming models for versions a3 and up
    return mn(getattr(_C.MODEL, model_name), causal=casual, pretrained=True)


def get_label_map(wanted_classes_path, kinetics_600_path):
    wanted_classes = read_labels_from_csv(wanted_classes_path)
    kinetics_classes = read_labels_from_csv(kinetics_600_path)
    label_map = {}
    for wanted_class in wanted_classes:
        try:
            index = kinetics_classes.index(wanted_class)
            label_map[index] = wanted_class
        except ValueError:
            print(f"Class {wanted_class} not found in {kinetics_600_path}")
    return label_map


def read_labels_from_csv(classes_path):
    labels = pd.read_csv(classes_path).iloc[:, -1].tolist()
    labels = [label.strip() for label in labels]
    return labels


def find_class_index(class_name, class_list):
    try:
        index = class_list.index(class_name)
        return index
    except ValueError:
        return -1


def load_kinetics(kinetics_path=KINETICS_PATH):
    if not os.path.exists(kinetics_path):
        url = 'https://gist.githubusercontent.com/willprice/f19da185c9c5f32847134b87c1960769/raw/9dc94028ecced572f302225c49fcdee2f3d748d8/kinetics_600_labels.csv'

        # Read the CSV file containing the class labels
        df = pd.read_csv(url)

        # save localy
        df.to_csv(kinetics_path)
    else:
        df = pd.read_csv(kinetics_path)
    dict_out = {idx: label for idx, label in enumerate(df.iloc[:, -1])}
    return dict_out
