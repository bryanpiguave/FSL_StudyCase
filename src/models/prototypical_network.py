import torch
import torch.nn as nn
import numpy as np

class PrototypicalNetwork(nn.Module):
    """
    Bootstrapped Prototypical Network for few-shot learning.
    
    This implementation creates multiple prototypes per class using bootstrapping,
    which improves robustness and performance compared to standard prototypical networks.
    
    Args:
        num_bootstrap_samples (int): Number of bootstrap samples to create per class
        bootstrap_ratio (float): Ratio of support samples to use in bootstrap (should be 1.0 to use all samples)
    """
    
    def __init__(self, num_bootstrap_samples=5, bootstrap_ratio=1.0):
        super(PrototypicalNetwork, self).__init__()
        self.prototypes = None
        self.num_bootstrap_samples = num_bootstrap_samples
        self.bootstrap_ratio = bootstrap_ratio
        self._bootstrap_cache = {}  # Cache for bootstrap indices
        
    def forward(self, support_set_repr, support_labels, query_set_repr):
        """
        Forward pass for the Bootstrapped Prototypical Network.
        Creates multiple prototypes per class using bootstrapping.
        Optimized for speed.
        
        Args:
            support_set_repr (torch.Tensor): Support set representations [N_support, D]
            support_labels (torch.Tensor): Support set labels [N_support]
            query_set_repr (torch.Tensor): Query set representations [N_query, D]
            
        Returns:
            torch.Tensor: Logits for query samples [N_query, num_classes]
        """
        if self.prototypes is None: 
            self.prototypes = support_set_repr
        else:
            self.prototypes = self.prototypes.mean(dim=0)
            
        # Separate the support set into classes - compute once
        support_set_classes = support_labels.unique()
        class_indices_dict = {}
        for class_label in support_set_classes:
            class_indices = (support_labels == class_label).nonzero(as_tuple=True)[0]
            class_indices_dict[class_label.item()] = class_indices
        
        all_bootstrap_logits = []
        
        # Pre-compute bootstrap indices for efficiency
        bootstrap_indices_cache = {}
        for class_label in support_set_classes:
            class_label_item = class_label.item()
            class_repr = support_set_repr[class_indices_dict[class_label_item]]
            num_samples = len(class_repr)
            
            # Generate all bootstrap indices for this class at once
            bootstrap_indices = torch.randint(0, num_samples, (self.num_bootstrap_samples, num_samples))
            bootstrap_indices_cache[class_label_item] = bootstrap_indices
        
        for bootstrap_idx in range(self.num_bootstrap_samples):
            bootstrap_class_prototypes = []
            
            for class_label in support_set_classes:
                class_label_item = class_label.item()
                class_indices = class_indices_dict[class_label_item]
                class_repr = support_set_repr[class_indices]
                
                # Use pre-computed bootstrap indices
                bootstrap_indices = bootstrap_indices_cache[class_label_item][bootstrap_idx]
                bootstrap_class_repr = class_repr[bootstrap_indices]
                
                # Compute prototype as mean of bootstrap sample
                class_prototype = bootstrap_class_repr.mean(dim=0)
                bootstrap_class_prototypes.append(class_prototype)
            
            bootstrap_class_prototypes = torch.stack(bootstrap_class_prototypes)
            
            # Compute distances for this bootstrap sample
            dists = torch.cdist(query_set_repr, bootstrap_class_prototypes, p=2)
            bootstrap_logits = -dists
            all_bootstrap_logits.append(bootstrap_logits)
        
        # Aggregate predictions from all bootstrap samples
        all_bootstrap_logits = torch.stack(all_bootstrap_logits, dim=0)
        aggregated_logits = all_bootstrap_logits.mean(dim=0)
        
        return aggregated_logits
        
    def predict(self, support_set_repr, support_labels, query_set_repr):
        """
        Predict the labels for the query set.
        
        Args:
            support_set_repr (torch.Tensor): Support set representations [N_support, D]
            support_labels (torch.Tensor): Support set labels [N_support]
            query_set_repr (torch.Tensor): Query set representations [N_query, D]
            
        Returns:
            torch.Tensor: Predicted labels for query samples [N_query]
        """
        logits = self.forward(support_set_repr, support_labels, query_set_repr)
        return logits.argmax(dim=1)
    
    def reset_prototypes(self):
        """
        Reset the stored prototypes. Useful when switching between different support sets.
        """
        self.prototypes = None 