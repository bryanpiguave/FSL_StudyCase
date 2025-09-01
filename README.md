# FSL Study Case: Molecular Toxicity Prediction with Prototypical Networks

This repository contains a comprehensive implementation of **Few-Shot Learning (FSL)** for molecular toxicity prediction using the Tox21 dataset. The project demonstrates how prototypical networks can be effectively applied to chemical compound classification tasks with limited labeled data.

## 🧪 Project Overview

The goal of this study is to predict molecular toxicity properties using few-shot learning approaches, specifically implementing a **Bootstrapped Prototypical Network** that can learn from very few examples per class. This is particularly valuable in drug discovery and chemical safety assessment where labeled data is often scarce.

## 🏗️ Architecture

### Core Components

- **Prototypical Network**: A few-shot learning model that creates class prototypes from support examples
- **ECFP Encoder**: Molecular fingerprint encoder using Extended Connectivity Fingerprints
- **Meta-Training Framework**: Training pipeline optimized for few-shot learning scenarios
- **Tox21 Dataset Integration**: Complete pipeline for the Tox21 toxicity prediction benchmark

### Key Features

- **Bootstrapped Prototypes**: Multiple prototypes per class for improved robustness
- **Molecular Fingerprinting**: ECFP-based molecular representation
- **Meta-Learning**: Training strategy that mimics few-shot scenarios
- **Comprehensive Evaluation**: Multi-run testing with statistical significance

## 📁 Repository Structure

```
FSL_StudyCase/
├── src/
│   ├── models/
│   │   └── prototypical_network.py    # Bootstrapped prototypical network implementation
│   ├── data/
│   │   └── efcp_utils.py              # ECFP fingerprint generation utilities
│   └── utils/                         # Training and evaluation utilities
├── metatraining/
│   └── protonet_metatrain.py         # Main meta-training script
├── configs/
│   └── dataset.yaml                   # Dataset configuration for Tox21
├── dataset/
│   └── tox21.csv                     # Tox21 dataset
├── requirements.txt                   # Python dependencies
└── Tox21_prototypical_network_best.pth  # Pre-trained model weights
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd FSL_StudyCase

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Meta-Training

```bash
cd metatraining
python protonet_metatrain.py \
    --shots 5 \
    --num_eval_runs 10 \
    --epochs 100 \
    --learning_rate 0.001
```

### 3. Evaluate Pre-trained Model

```bash
python protonet_metatrain.py \
    --model_path ../Tox21_prototypical_network_best.pth \
    --evaluate_only \
    --shots 5 \
    --num_eval_runs 10
```

## 🔬 Technical Details

### Prototypical Network

The core model implements a **bootstrapped prototypical network** that:

- Creates multiple prototypes per class using bootstrap sampling
- Improves robustness compared to standard prototypical networks
- Optimizes computational efficiency with pre-computed bootstrap indices
- Aggregates predictions from multiple bootstrap samples

### Molecular Representation

- **ECFP Fingerprints**: 2048-bit Extended Connectivity Fingerprints (radius=2)
- **Encoder Architecture**: Multi-layer perceptron with ReLU activation and dropout
- **Input Processing**: Handles invalid SMILES gracefully with zero-padding

### Training Strategy

- **Meta-Training**: Simulates few-shot scenarios during training
- **Task Sampling**: Randomly samples tasks from Tox21 dataset
- **Support/Query Split**: Dynamic creation of support and query sets
- **Multi-Run Evaluation**: Multiple evaluation runs for statistical significance

## 📊 Dataset: Tox21

The Tox21 (Toxicology in the 21st Century) dataset contains:

- **12 toxicity endpoints** for chemical compounds
- **7,831 compounds** with SMILES representations
- **Binary classification** tasks (toxic/non-toxic)
- **Train/Test Split**: 9 training tasks, 3 held-out test tasks

### Available Tasks

**Training Tasks:**
- NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase
- NR-ER, NR-ER-LBD, NR-PPAR-gamma
- SR-ARE, SR-ATAD5

**Test Tasks:**
- SR-HSE, SR-MMP, SR-p53

## ⚙️ Configuration

### Training Parameters

- `--shots`: Number of support examples per class (default: 5)
- `--num_eval_runs`: Number of evaluation runs per task (default: 10)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for optimization (default: 0.001)
- `--bootstrap_samples`: Number of bootstrap samples per class (default: 5)

### Model Parameters

- **ECFP Dimension**: 2048 bits
- **Hidden Dimension**: 128 units
- **Output Dimension**: 128 units
- **Dropout Rate**: 0.1

## 📈 Performance

The model achieves competitive performance on Tox21 few-shot learning tasks:

- **5-shot learning** with limited labeled data
- **Robust evaluation** through multiple bootstrap samples
- **Statistical significance** through multi-run testing
- **Efficient inference** with optimized prototype computation

## 🔧 Dependencies

### Core ML Libraries
- PyTorch ≥2.4.1
- NumPy ≥2.3.0
- scikit-learn ≥1.0.0

### Chemistry Libraries
- RDKit ≥2025.03.5
- Mordred Community (full)

### Additional Tools
- Transformers ≥4.20.0
- WandB ≥0.12.0 (experiment tracking)
- Plotly ≥5.0.0 (visualization)

## 📝 Usage Examples

### Basic Training

```python
from src.models.prototypical_network import PrototypicalNetwork
from src.data.efcp_utils import generate_ecfp_fingerprints

# Initialize model
model = PrototypicalNetwork(num_bootstrap_samples=5)

# Generate molecular fingerprints
smiles_list = ["CCO", "CCCO", "CCCC"]
fingerprints = generate_ecfp_fingerprints(smiles_list)

# Use in few-shot learning scenario
# ... (see training script for complete examples)
```

### Custom Configuration

```yaml
# configs/custom_dataset.yaml
CustomDataset:
  train_tasks:
    - "task1"
    - "task2"
  test_tasks:
    - "task3"
  split_test_tasks_path: "path/to/splits.json"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 References

- **Prototypical Networks**: Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning
- **Tox21 Dataset**: Huang, R., et al. (2016). Tox21 Challenge to Build Predictive Models of Nuclear Receptor and Stress Response Pathways
- **ECFP Fingerprints**: Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
---

**Note**: This is a research implementation for educational and research purposes. For production use, additional validation and testing is recommended.
