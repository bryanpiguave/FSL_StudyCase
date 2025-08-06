#!/usr/bin/env python3
"""
Shared utilities for training scripts.
Contains common functions used across different prototypical network training scripts.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score


def save_metrics(output_dir, dataset_name, train_losses, test_losses, test_aucs, episode, model_type=""):
    """
    Save metrics to CSV files in real-time.
    
    Args:
        output_dir (str): Directory to save the metrics
        dataset_name (str): Name of the dataset
        train_losses (dict): Dictionary of training losses per task
        test_losses (dict): Dictionary of test losses per task
        test_aucs (dict): Dictionary of test AUCs per task
        episode (int): Current episode number
        model_type (str): Type of model (e.g., "bimodal", "llm_only", etc.) for filename prefix
    """
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine filename prefix based on model type
    prefix = f"{model_type}_" if model_type else ""
    
    # Save train losses
    train_losses_df = pd.DataFrame(train_losses)
    train_losses_df.to_csv(f"{output_dir}/{dataset_name}_{prefix}prototypical_network_losses.csv", index=False)
    
    # Save test losses
    test_losses_df = pd.DataFrame(test_losses)
    test_losses_df.to_csv(f"{output_dir}/{dataset_name}_{prefix}prototypical_network_test_losses.csv", index=False)
    
    # Save test AUCs
    test_aucs_df = pd.DataFrame(test_aucs)
    test_aucs_df.to_csv(f"{output_dir}/{dataset_name}_{prefix}prototypical_network_test_aucs.csv", index=False)
    
    # Save episode info
    episode_info = {
        'episode': episode,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    episode_df = pd.DataFrame([episode_info])
    episode_df.to_csv(f"{output_dir}/{dataset_name}_{prefix}prototypical_network_episode_info.csv", index=False)


def save_final_results(output_dir, dataset_name, final_results, model_type=""):
    """
    Save final evaluation results to CSV file.
    
    Args:
        output_dir (str): Directory to save the results
        dataset_name (str): Name of the dataset
        final_results (dict): Dictionary containing evaluation statistics
        model_type (str): Type of model for filename prefix
    """
    # Create necessary directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine filename prefix based on model type
    prefix = f"{model_type}_" if model_type else ""
    
    # Prepare data for CSV
    rows = []
    for task_id, stats in final_results.items():
        row = {
            'task_id': task_id,
            'auc_mean': stats['auc_mean'],
            'auc_std': stats['auc_std'],
            'auc_min': stats['auc_min'],
            'auc_max': stats['auc_max'],
            'loss_mean': stats['loss_mean'],
            'loss_std': stats['loss_std'],
            'loss_min': stats['loss_min'],
            'loss_max': stats['loss_max']
        }
        rows.append(row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/{dataset_name}_{prefix}final_evaluation_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Final evaluation results saved to: {csv_path}")
    
    return csv_path


def final_evaluation_generic(model, prototypical_network, test_tasks, df, args, device, criterion, 
                           data_processor_func, num_runs=30):
    """
    Generic final evaluation function that can work with different data processing approaches.
    
    Args:
        model: The trained model
        prototypical_network: Prototypical network instance
        test_tasks: List of test task names
        df: DataFrame containing the dataset
        args: Arguments object
        device: Device to run evaluation on
        criterion: Loss function
        data_processor_func: Function that processes data for the specific model type
        num_runs: Number of evaluation runs per task
    
    Returns:
        dict: Final evaluation results with statistics
    """
    model.eval()
    task_aucs_all_runs = {task: [] for task in test_tasks}
    task_losses_all_runs = {task: [] for task in test_tasks}
    
    print(f"Running final evaluation with {num_runs} runs per task...")
    
    with torch.no_grad():
        for task_id in tqdm(test_tasks, desc="Final evaluation tasks"):
            print(f"Evaluating task {task_id} with {num_runs} runs...")
            
            for run in range(num_runs):
                # Use the provided data processor function
                support_data, query_data, support_labels, query_labels = data_processor_func(
                    df, task_id, args, run
                )
                
                # Move data to device
                support_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in support_data.items()}
                query_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in query_data.items()}
                support_labels = support_labels.to(device)
                query_labels = query_labels.to(device)
                
                # Forward pass
                support_repr = model(**support_data)
                query_repr = model(**query_data)
                logits = prototypical_network(support_repr, support_labels, query_repr)
                
                # Compute AUC
                probs = torch.softmax(logits, dim=1)
                positive_probs = probs[:, 1]
                auc = roc_auc_score(query_labels.cpu().numpy(), positive_probs.cpu().numpy())
                task_aucs_all_runs[task_id].append(auc)
                
                # Compute loss
                loss = criterion(logits, query_labels)
                task_losses_all_runs[task_id].append(loss.item())
                
                if (run + 1) % 10 == 0:
                    print(f"  Run {run+1}: AUC = {auc:.4f}, Loss = {loss.item():.4f}")
    
    # Calculate statistics
    final_results = {}
    for task_id in test_tasks:
        aucs = task_aucs_all_runs[task_id]
        losses = task_losses_all_runs[task_id]
        
        final_results[task_id] = {
            'auc_mean': np.mean(aucs),
            'auc_std': np.std(aucs),
            'loss_mean': np.mean(losses),
            'loss_std': np.std(losses),
            'auc_min': np.min(aucs),
            'auc_max': np.max(aucs),
            'loss_min': np.min(losses),
            'loss_max': np.max(losses)
        }
        
        print(f"Task {task_id}:")
        print(f"  AUC: {final_results[task_id]['auc_mean']:.4f} ± {final_results[task_id]['auc_std']:.4f}")
        print(f"  Loss: {final_results[task_id]['loss_mean']:.4f} ± {final_results[task_id]['loss_std']:.4f}")
    
    # Calculate overall statistics
    all_aucs = []
    all_losses = []
    for task_id in test_tasks:
        all_aucs.extend(task_aucs_all_runs[task_id])
        all_losses.extend(task_losses_all_runs[task_id])
    
    final_results['Overall'] = {
        'auc_mean': np.mean(all_aucs),
        'auc_std': np.std(all_aucs),
        'loss_mean': np.mean(all_losses),
        'loss_std': np.std(all_losses),
        'auc_min': np.min(all_aucs),
        'auc_max': np.max(all_aucs),
        'loss_min': np.min(all_losses),
        'loss_max': np.max(all_losses)
    }
    
    print(f"\nOverall Results:")
    print(f"  AUC: {final_results['Overall']['auc_mean']:.4f} ± {final_results['Overall']['auc_std']:.4f}")
    print(f"  Loss: {final_results['Overall']['loss_mean']:.4f} ± {final_results['Overall']['loss_std']:.4f}")
    
    return final_results 