import argparse
import os
import torch
import numpy as np
import random
import dgl
from utils import get_logger
from trainer import train

from graph_builder import kddcup_graph
from xuetangx_graph_builder import XuetangXGraph

def set_seed(seed):
    """
    设置全局随机种子，确保实验可复现 (用于 T-test)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # DGL 随机种子
    dgl.random.seed(seed)

def main(args):
    """
    主函数：负责环境设置、日志初始化、数据加载以及训练入口的分发。
    """
    # 1. [新增] 设置随机种子 (最先执行)
    set_seed(args.seed)

    # 2. 设置日志记录器
    # [修改] 文件名包含 seed，便于区分不同随机种子的实验结果
    log_filename = f"{args.dataset}_{args.model}_lay{args.num_layers}_win{args.window_size}_seed{args.seed}.log"
    log_path = os.path.join(args.log_dir, log_filename)
    
    # 获取 logger 实例
    logger = get_logger(name=f"{args.dataset}_{args.model}", path=log_path)
    
    logger.info('=' * 30)
    logger.info('--- Experiment Arguments ---')
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('=' * 30 + '\n')

    # 3. 加载数据集
    dataset = None
    
    if args.dataset == 'kddcup':
        logger.info(f"Loading KDDCUP dataset with window_size={args.window_size}...")
        dataset = kddcup_graph(
            subset_ratio=args.subset_ratio, 
            window_size=args.window_size
        )
        
    elif args.dataset == 'xuetangx':
        logger.info(f"Loading XuetangX dataset (Subset Ratio: {args.subset_ratio})...")
        
        # 定义数据路径
        raw_data_path = 'data/xuetangx' 
        cache_path = 'data/xuetangx/cache'
        
        # 确保缓存目录存在
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
            
        # 实例化 XuetangXGraph
        # 注意: window_size 在 XuetangXGraph 内部逻辑中处理
        dataset = XuetangXGraph(
            raw_dir=raw_data_path,
            save_dir=cache_path,
            subset_ratio=args.subset_ratio,
            force_reload=False  # 如果修改了构图逻辑，请改为 True 或手动删除 cache 文件夹
        )
        
    else:
        logger.error(f"Dataset '{args.dataset}' is not supported. Please use 'kddcup' or 'xuetangx'.")
        return

    # 4. 启动训练流程
    # 根据模型名称决定调用哪个训练器
    
    if args.model == 'XGB':
        logger.info(">>> Mode: XGBoost Baseline Training")
        train_xgboost(args, dataset, logger)
        
    else:
        logger.info(f">>> Mode: Deep Learning Training ({args.model})")
        # 调用 PyTorch 训练流程 
        # 支持: Multi_MST_GCN, SASRec, Multi_HAN, SIG_Net, etc.
        train(args, dataset, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOOC Dropout Prediction Experiment Runner")

    # ==========================================
    # 1. 数据集与路径参数
    # ==========================================
    parser.add_argument("--dataset", type=str, default='xuetangx', 
                        choices=['kddcup', 'xuetangx'],
                        help="Dataset selection.")
    
    parser.add_argument("--log_dir", type=str, default='experiment_log_final/', 
                        help="Directory to save log files and visualizations.")
    
    parser.add_argument("--subset_ratio", type=float, default=1, 
                        help="Data sampling ratio (0.0-1.0). Use small value (e.g., 0.01) for debugging.")

    # ==========================================
    # 2. [关键修改] 随机种子 (用于 T-test)
    # ==========================================
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed for reproducibility. Run 0-4 for Statistical Test.")

    # ==========================================
    # 3. 模型选择与架构参数
    # ==========================================
    parser.add_argument("--model", type=str, default='Multi_MST_GCN', 
                        help="Model name. Options: 'Multi_MST_GCN', 'SASRec', 'Multi_HAN', 'SIG_Net', 'BiLSTM_Model', 'XGB'")
    
    parser.add_argument("--window_size", type=int, default=12, 
                        help="Window size for temporal graph snapshotting.")

    parser.add_argument("--hidden_dim", type=int, default=128, 
                        help="Hidden dimension size for embeddings/layers.")
    
    parser.add_argument("--num_layers", type=int, default=2, 
                        help="Number of layers for GNN/LSTM/CNN.")
    
    parser.add_argument("--dropout", type=float, default=0.3, 
                        help="Dropout probability.")
    
    parser.add_argument("--bi", action='store_true', default=True, 
                        help="Use bidirectional processing (for LSTM/RNN variants).")

    # [新增] Attention Heads 参数 (SASRec / Multi_HAN 需要)
    parser.add_argument("--num_heads", type=int, default=4, 
                        help="Number of attention heads for Transformer/GAT based models.")

    # ==========================================
    # 4. 训练超参数
    # ==========================================
    parser.add_argument("--lr", type=float, default=0.0001, 
                        help="Learning rate.")
    
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size.")
    
    parser.add_argument("-e", "--num_epochs", type=int, default=8, 
                        help="Maximum number of training epochs.")

    # ==========================================
    # 5. 系统环境参数
    # ==========================================
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], 
                        help="List of GPU IDs to use (e.g., --gpu 0 1).")
    
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of dataloader workers (0 for safest Windows/Linux compatibility).")
    
    # 解析参数
    args = parser.parse_args()
    
    # 自动创建日志目录
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # 执行主程序
    main(args)