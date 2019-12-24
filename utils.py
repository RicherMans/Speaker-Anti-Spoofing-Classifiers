#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import logging
import sys
from pprint import pformat

import numpy as np
import pandas as pd
import six
import sklearn.preprocessing as pre
import torch
import yaml

import augment


def parse_config_or_kwargs(config_file, **kwargs):
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    return dict(yaml_config, **kwargs)


def split_train_cv(data_frame: pd.DataFrame, frac: float = 0.9):
    """split_train_cv

    :param data_frame:
    :type data_frame: pd.DataFrame
    :param frac:
    :type frac: float
    """
    train_data = data_frame.sample(frac=frac, random_state=0)
    cv_data = data_frame[~data_frame.index.isin(train_data.index)]
    return train_data, cv_data


def parse_transforms(transform_list):
    """parse_transforms
    parses the config files transformation strings to coresponding methods

    :param transform_list: String list
    """
    transforms = []
    for trans in transform_list:
        if trans == 'noise':
            transforms.append(augment.GaussianNoise(0, 0.05))
        elif trans == 'roll':
            transforms.append(augment.Roll(0, 10))
        elif trans == 'freqmask':
            transforms.append(augment.FreqMask(1, 24))
        elif trans == 'timemask':
            transforms.append(augment.TimeMask(1, 60))
        elif trans == 'randompad':
            transforms.append(augment.RandomPad(value=0., padding=25))
    return torch.nn.Sequential(*transforms)


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict

    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def getfile_outlogger(outputfile):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(logging.INFO)
    stdlog = logging.StreamHandler(sys.stdout)
    stdlog.setFormatter(formatter)
    file_handler = logging.FileHandler(outputfile)
    file_handler.setFormatter(formatter)
    # Log to stdout
    logger.addHandler(file_handler)
    logger.addHandler(stdlog)
    return logger


def encode_labels(labels: pd.Series, encoder=None):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (many hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to be series"
    if isinstance(labels[0], six.string_types):
        # In case of using non processed strings, e.g., Vaccum, Speech
        label_array = labels.str.split(',').values.tolist()
    elif isinstance(labels[0], np.ndarray):
        # Encoder does not like to see numpy array
        label_array = [lab.tolist() for lab in labels]
    elif isinstance(labels[0], collections.Iterable):
        label_array = labels
    if not encoder:
        encoder = pre.LabelBinarizer()
        encoder.fit(label_array)
    labels_encoded = encoder.transform(label_array)
    return labels_encoded.astype(np.float32).tolist(), encoder
