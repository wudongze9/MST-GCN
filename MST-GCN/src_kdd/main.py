import argparse
from graph_builder import kddcup_graph
from utils import get_logger
from trainer import train
import os

def main(args):
    """
    Main function to set up logging, load data, and start training.
    """
    # Use command line argument as model name for log naming
    model_name_for_log = args.model 
    
    # Set up logger
    log_path = f"{args.log_dir}/{args.dataset}_{args.model}_layers{args.num_layers}_bs{args.batch_size}_drop{args.dropout}_win{args.window_size}_seed{args.seed}.log"
    logger = get_logger(name=f"{args.dataset}_{args.model}", path=log_path)
    
    logger.info('--- Training Arguments ---')
    logger.info(args)
    logger.info('--------------------------\n')

    # Load dataset based on args
    if args.dataset == 'kddcup':
        # =================================================================
        # *** Modification 2: Pass window_size argument ***
        # =================================================================
        dataset = kddcup_graph(
            subset_ratio=args.subset_ratio, 
            window_size=args.window_size
        )
    else:
        logger.error(f"Dataset '{args.dataset}' is not supported. Only 'kddcup' is available.")
        return

    # Call training function
    train(args, dataset, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-based Dropout Prediction Model Training")

    # --- Dataset & Log Parameters ---
    parser.add_argument("--dataset", type=str, default='kddcup', help="Dataset to use (default: 'kddcup')")
    parser.add_argument("--log_dir", type=str, default='experiment_log/', help="Directory to save log files")
    parser.add_argument("--subset_ratio", type=float, default=1, 
                        help="Ratio of data to use for the experiment (1.0 for full dataset, 0.01 for 1%)")

    # =================================================================
    # *** Modification 3: Add --window_size argument ***
    # =================================================================
    parser.add_argument("--window_size", type=int, default=10, 
                        help="Days per snapshot. Total days = window_size * 3. Default: 10 (30 days total)")

    # =================================================================
    # *** CRITICAL FIX: Add --seed argument ***
    # =================================================================
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed for reproducibility. Run 0-4 for Statistical Test.")

    # --- Model Architecture Parameters ---
    parser.add_argument("--model", type=str, default='Multi_MST_GCN', 
                        help="GNN model to use (e.g., Multi_MST_GCN)")

    parser.add_argument("--hidden_dim", type=int, default=128, help="Number of hidden units in GNN layers")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--bi", action='store_true', default=True, help="Use Bi-directional LSTM/Transformer (legacy)")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("-e", "--num_epochs", type=int, default=8, help="Number of training epochs")

    # --- System & Execution Parameters ---
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help="List of GPU IDs to use (e.g., 0, 0 1)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of worker processes for data loading")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # Start main execution
    main(args)