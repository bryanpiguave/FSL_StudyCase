import sys
import os
import warnings

# Set tokenizers parallelism to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress the specific FutureWarning from transformers
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*", category=FutureWarning)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import yaml 
import argparse
import pandas as pd
from src.models.prototypical_network import PrototypicalNetwork
from src.utils.training_utils import save_metrics, save_final_results

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from collections import defaultdict
import json
import numpy as np
import random
# Generate ECFP fingerprints (simplified)
from rdkit import Chem
from rdkit.Chem import AllChem
                
def smiles_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))
                
class ECFPEncoder(nn.Module):
    """
    Simple ECFP encoder for prototypical network.
    """
    def __init__(self, ecfp_dim=2048, hidden_dim=128, output_dim=128):
        super(ECFPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ecfp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, ecfp):
        return self.encoder(ecfp)

# PrototypicalNetwork class is now imported from src.models.prototypical_network

def evaluate_model(model, prototypical_network, test_tasks, df, args, device, criterion):
    """
    Evaluate the model on meta-test tasks and return average AUC and losses.
    Runs each task multiple times and averages the results.
    """
    model.eval()
    task_aucs_all_runs = {task: [] for task in test_tasks}
    task_losses_all_runs = {task: [] for task in test_tasks}
    
    with torch.no_grad():
        for task_id in tqdm(test_tasks, desc="Evaluating tasks"):
            print(f"Evaluating task {task_id} with {args.num_eval_runs} runs...")
            
            for run in range(args.num_eval_runs):
                # Create simple dataset for this task
                task_data = df[df[task_id] == 1]  # Positive samples
                task_data_neg = df[df[task_id] == 0]  # Negative samples
                
                # Sample support and query sets
                n_shot = args.shots
                n_query = 32
                
                # Sample support set
                pos_samples = task_data.sample(min(n_shot, len(task_data)))
                neg_samples = task_data_neg.sample(min(n_shot, len(task_data_neg)))
                support_data = pd.concat([pos_samples, neg_samples])
                
                # Sample query set
                remaining_pos = task_data.drop(pos_samples.index)
                remaining_neg = task_data_neg.drop(neg_samples.index)
                # Evaluate on all remaining samples
                query_pos = remaining_pos
                query_neg = remaining_neg
                query_data = pd.concat([query_pos, query_neg])
                
                # Generate ECFP fingerprints (simplified)
                from rdkit import Chem
                from rdkit.Chem import AllChem
                
                def smiles_to_ecfp(smiles, radius=2, nBits=2048):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return np.zeros(nBits)
                    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))
                
                # Support set ECFP
                support_ecfp = torch.tensor([smiles_to_ecfp(s) for s in support_data['smiles']], dtype=torch.float32)
                support_labels = torch.tensor([1 if i < len(pos_samples) else 0 for i in range(len(support_data))], dtype=torch.long)
                
                # Query set ECFP
                query_ecfp = torch.tensor([smiles_to_ecfp(s) for s in query_data['smiles']], dtype=torch.float32)
                query_labels = torch.tensor([1 if i < len(query_pos) else 0 for i in range(len(query_data))], dtype=torch.long)
                
                # Move to device
                support_ecfp = support_ecfp.to(device)
                support_labels = support_labels.to(device)
                query_ecfp = query_ecfp.to(device)
                query_labels = query_labels.to(device)
                
                # Forward pass
                support_repr = model(support_ecfp)
                query_repr = model(query_ecfp)
                logits = prototypical_network(support_repr, support_labels, query_repr)
                
                # Compute AUC
                probs = torch.softmax(logits, dim=1)
                positive_probs = probs[:, 1]
                auc = roc_auc_score(query_labels.cpu().numpy(), positive_probs.cpu().numpy())
                task_aucs_all_runs[task_id].append(auc)
                
                # Compute loss
                loss = criterion(logits, query_labels)
                task_losses_all_runs[task_id].append(loss.item())
                
                print(f"  Run {run+1}: AUC = {auc:.4f}, Loss = {loss.item():.4f}")
    
    # Average results across runs
    task_aucs = {}
    task_losses = {}
    average_auc = 0
    
    for task_id in test_tasks:
        task_aucs[task_id] = np.mean(task_aucs_all_runs[task_id])
        task_losses[task_id] = np.mean(task_losses_all_runs[task_id])
        average_auc += task_aucs[task_id]
        print(f"Task {task_id}: Average AUC = {task_aucs[task_id]:.4f} ± {np.std(task_aucs_all_runs[task_id]):.4f}")
    
    average_auc /= len(test_tasks)
    return average_auc, task_aucs, task_losses


def final_evaluation(model, prototypical_network, test_tasks, df, args, device, criterion, num_runs=30):
    """
    Final evaluation of the best model with multiple runs to get reliable statistics.
    """
    model.eval()
    task_aucs_all_runs = {task: [] for task in test_tasks}
    task_losses_all_runs = {task: [] for task in test_tasks}
    
    print(f"Running final evaluation with {num_runs} runs per task...")
    
    with torch.no_grad():
        for task_id in tqdm(test_tasks, desc="Final evaluation tasks"):
            print(f"Evaluating task {task_id} with {num_runs} runs...")
            
            for run in range(num_runs):
                # Create simple dataset for this task
                task_data = df[df[task_id] == 1]  # Positive samples
                task_data_neg = df[df[task_id] == 0]  # Negative samples
                
                # Sample support and query sets
                n_shot = args.shots
                n_query = 32
                
                # Sample support set
                pos_samples = task_data.sample(min(n_shot, len(task_data)))
                neg_samples = task_data_neg.sample(min(n_shot, len(task_data_neg)))
                support_data = pd.concat([pos_samples, neg_samples])
                
                # Sample query set
                remaining_pos = task_data.drop(pos_samples.index)
                remaining_neg = task_data_neg.drop(neg_samples.index)
                query_pos = remaining_pos.sample(min(n_query//2, len(remaining_pos)))
                query_neg = remaining_neg.sample(min(n_query//2, len(remaining_neg)))
                query_data = pd.concat([query_pos, query_neg])
                
              
                # Support set ECFP
                support_ecfp = torch.tensor([smiles_to_ecfp(s) for s in support_data['smiles']], dtype=torch.float32)
                support_labels = torch.tensor([1 if i < len(pos_samples) else 0 for i in range(len(support_data))], dtype=torch.long)
                
                # Query set ECFP
                query_ecfp = torch.tensor([smiles_to_ecfp(s) for s in query_data['smiles']], dtype=torch.float32)
                query_labels = torch.tensor([1 if i < len(query_pos) else 0 for i in range(len(query_data))], dtype=torch.long)
                
                # Move to device
                support_ecfp = support_ecfp.to(device)
                support_labels = support_labels.to(device)
                query_ecfp = query_ecfp.to(device)
                query_labels = query_labels.to(device)
                
                # Forward pass
                support_repr = model(support_ecfp)
                query_repr = model(query_ecfp)
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



def main():
    print("Starting ECFP-only meta-training...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["Tox21", "SIDER", "MUV"], required=True)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--ecfp_bits", type=int, default=2048)
    parser.add_argument('--output_dir', type=str, default='logs_ecfp_only')
    parser.add_argument('--model_dir', type=str, default='models_ecfp_only')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--eval_freq', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dim', type=int, default=128, help='Output dimension of the model for the prototypical network')
    parser.add_argument('--num_bootstrap_samples', type=int, default=5, help='Number of bootstrap samples')
    parser.add_argument('--bootstrap_ratio', type=float, default=1.0, help='Ratio of support samples to use in bootstrap (should be 1.0 to use all samples)')
    parser.add_argument('--eval_batch_size_multiplier', type=int, default=2, help='Multiplier for evaluation batch size to speed up evaluation')
    parser.add_argument('--num_eval_runs', type=int, default=5, help='Number of evaluation runs per task to average results')
    args = parser.parse_args()

    with open(f"configs/dataset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)
        
    dataset_path = f'datasets/{args.dataset_name.lower()}.csv'
    dataset_config = dataset_config[args.dataset_name]
    test_tasks = dataset_config['test_tasks']
    train_tasks = dataset_config['train_tasks']
    df = pd.read_csv(dataset_path)
    
    print('Dataset loaded')

    # Create ECFP-only model
    model = ECFPEncoder(
        ecfp_dim=args.ecfp_bits,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim
    )
    print(100*'-')
    print(model)
    print('Model created')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prototypical_network = PrototypicalNetwork(
        num_bootstrap_samples=args.num_bootstrap_samples,
        bootstrap_ratio=args.bootstrap_ratio
    )
    prototypical_network.to(device)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()

    # Tracking metrics
    train_losses = {task: [] for task in train_tasks}
    test_losses = {task: [] for task in test_tasks}
    test_aucs = {task: [] for task in test_tasks}
    test_aucs['Average_AUC'] = []
    
    # Early stopping variables
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    print('Starting meta-training...')
    print(f'Bootstrapping configuration: {args.num_bootstrap_samples} samples, {args.bootstrap_ratio} ratio')
    
    for episode in range(args.episodes):
        model.train()
        print(f"Episode {episode+1}/{args.episodes}")
        
        # Train the model
        for i in tqdm(range(len(train_tasks)), desc="Training tasks"):
            task_id = train_tasks[i]
            
            # Sample data for this task
            task_data = df[df[task_id] == 1]  # Positive samples
            task_data_neg = df[df[task_id] == 0]  # Negative samples
            
            # Sample support and query sets
            n_shot = args.shots
            n_query = 25
            
            # Sample support set
            pos_samples = task_data.sample(min(n_shot, len(task_data)))
            neg_samples = task_data_neg.sample(min(n_shot, len(task_data_neg)))
            support_data = pd.concat([pos_samples, neg_samples])
            
            # Sample query set
            remaining_pos = task_data.drop(pos_samples.index)
            remaining_neg = task_data_neg.drop(neg_samples.index)
            query_pos = remaining_pos.sample(min(n_query//2, len(remaining_pos)))
            query_neg = remaining_neg.sample(min(n_query//2, len(remaining_neg)))
            query_data = pd.concat([query_pos, query_neg])
                        
            # Support set ECFP
            support_ecfp = torch.tensor([smiles_to_ecfp(s) for s in support_data['smiles']], dtype=torch.float32)
            support_labels = torch.tensor([1 if i < len(pos_samples) else 0 for i in range(len(support_data))], dtype=torch.long)
            
            # Query set ECFP
            query_ecfp = torch.tensor([smiles_to_ecfp(s) for s in query_data['smiles']], dtype=torch.float32)
            query_labels = torch.tensor([1 if i < len(query_pos) else 0 for i in range(len(query_data))], dtype=torch.long)
            
            # Move to device
            support_ecfp = support_ecfp.to(device)
            support_labels = support_labels.to(device)
            query_ecfp = query_ecfp.to(device)
            query_labels = query_labels.to(device)
            
            # Forward pass
            support_repr = model(support_ecfp)
            query_repr = model(query_ecfp)
            logits = prototypical_network(support_repr, support_labels, query_repr)

            # Compute the loss
            loss = criterion(logits, query_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses[task_id].append(loss.item())
        
        # Evaluate the model
        if episode % args.eval_freq == 0:   
            print("Meta-testing...")
            average_auc, task_aucs, task_losses = evaluate_model(model, prototypical_network, test_tasks, df, args, device, criterion)
            # Update metrics
            test_aucs['Average_AUC'].append(average_auc)
            for task_id, auc in task_aucs.items():
                test_aucs[task_id].append(auc)
            for task_id, loss in task_losses.items():
                test_losses[task_id].append(loss)
            print(f"Episode {episode+1}: Average AUC = {average_auc:.4f}")
            for task_id, auc in task_aucs.items():
                print(f"  Task {task_id}: AUC = {auc:.4f} | Loss = {task_losses[task_id]:.4f}")
            # Save metrics in real-time
            save_metrics(args.output_dir, args.dataset_name, train_losses, test_losses, test_aucs, episode, "")
            # Early stopping check
            if average_auc > best_auc:
                best_auc = average_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"New best AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} evaluations. Best AUC: {best_auc:.4f}")
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {episode+1} episodes")
                break
    
    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, f"{args.model_dir}/{args.dataset_name}_prototypical_network_best.pth")
        print(f"Best model saved with AUC: {best_auc:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), f"{args.model_dir}/{args.dataset_name}_prototypical_network_final.pth")
    
    # Final metrics save
    save_metrics(args.output_dir, args.dataset_name, train_losses, test_losses, test_aucs, episode, "ecfp_only")
    
    # Run final evaluation on the best model
    if best_model_state is not None:
        print("Loading best model for final evaluation...")
        model.load_state_dict(best_model_state)
        final_results = final_evaluation(model, prototypical_network, test_tasks, df, args, device, criterion)
        save_final_results(args.output_dir, args.dataset_name, final_results, "")
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    return

if __name__ == "__main__":
    main() 