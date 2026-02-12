# utils.py
import dgl
import pandas as pd
import os
import torch
import logging

def collate_data(data):
    
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
    # 确保目录存在
    log_dir = os.path.dirname(path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
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

def xuetangx_load():
    """
    Loads all necessary data files for the XuetangX dataset.
    Assumption: Data is stored in 'data/xuetangx' with specific filenames.
    """
    path = 'data/xuetangx'
    
    # 检查路径是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"XuetangX data directory not found at: {path}")

    # 加载 User Profile
    # 包含: user_id, gender, education, birth
    user_profile = pd.read_csv(os.path.join(path, 'user_profile.csv'))

    # 加载 Course Information
    # 包含: id, course_id, start, end, course_type, category
    course_info = pd.read_csv(os.path.join(path, 'course_info.csv'))

    # 加载 Logs
    # 包含: enroll_id, username, course_id, session_id, action, object, time
    # 这里我们加载 train 和 test，如果需要可以合并
    train_log = pd.read_csv(os.path.join(path, 'train_log.csv'))
    # 如果存在 test_log，也可以加载
    if os.path.exists(os.path.join(path, 'test_log.csv')):
        test_log = pd.read_csv(os.path.join(path, 'test_log.csv'))
        log_all = pd.concat([train_log, test_log], ignore_index=True)
    else:
        log_all = train_log

    # 加载 Truth (Labels)
    # 包含: enroll_id, truth
    truth_train = pd.read_csv(os.path.join(path, 'train_truth.csv'))
    if os.path.exists(os.path.join(path, 'test_truth.csv')):
        truth_test = pd.read_csv(os.path.join(path, 'test_truth.csv'))
        truth_all = pd.concat([truth_train, truth_test], ignore_index=True)
    else:
        truth_all = truth_train

    # 返回所有相关 DataFrame
    return log_all, user_profile, course_info, truth_all

