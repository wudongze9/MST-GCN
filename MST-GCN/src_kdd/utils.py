# utils.py
import dgl
import pandas as pd
import os
import torch
import logging

def collate_data(data):
    """
    Collate function for DataLoader. Batches graphs and stacks labels.
    """
    subgraph_list1, subgraph_list2, subgraph_list3, label_list = map(list, zip(*data))
    
    subgraph1 = dgl.batch(subgraph_list1)
    subgraph2 = dgl.batch(subgraph_list2)
    subgraph3 = dgl.batch(subgraph_list3)
    g_label = torch.stack(label_list)
    
    return subgraph1, subgraph2, subgraph3, g_label
    
def get_logger(name, path):
    """
    Initializes and returns a logger instance.
    """
    logger = logging.getLogger(name)
    
    if len(logger.handlers) > 0:
        return logger # Logger already exists

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    # Console handler for debugging
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler for INFO level logs
    file_handler = logging.FileHandler(filename=path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def kddcup_load():
    """
    Loads all necessary data files for the KDD Cup 2015 dataset.
    """
    path = 'data/kddcup15'

    course_date = pd.read_csv(os.path.join(path, 'date.csv'))
    course_info = pd.read_csv(os.path.join(path, 'object.csv'))
    enrollment_train = pd.read_csv(os.path.join(path, 'enrollment_train.csv'))
    enrollment_test = pd.read_csv(os.path.join(path, 'enrollment_test.csv'))
    log_train = pd.read_csv(os.path.join(path, 'log_train.csv'))
    log_test = pd.read_csv(os.path.join(path, 'log_test.csv'))
    truth_train = pd.read_csv(os.path.join(path, 'truth_train.csv'), header=None, names=['enrollment_id', 'truth'])
    truth_test = pd.read_csv(os.path.join(path, 'truth_test.csv'), header=None, names=['enrollment_id', 'truth'])
    
    truth_all = pd.concat([truth_train, truth_test])
    log_all = pd.concat([log_train, log_test])
    enrollment_all = pd.concat([enrollment_train, enrollment_test])

    return course_date, course_info, enrollment_train, enrollment_test, log_train, log_test, truth_train, truth_test, truth_all, log_all, enrollment_all
