Transformer Model for If-Statement Prediction
A complete implementation of a Transformer-based model that predicts if-statement conditions in Python code using pre-training and fine-tuning.
🎯 Project Goals

Pre-train a Transformer model on Python code using Masked Language Modeling (MLM)
Fine-tune the model to predict if-statement conditions
Use custom-trained tokenizer (no pre-trained tokenizers allowed)
Achieve high accuracy on if-condition prediction

📋 Requirements
Dataset Requirements

Pre-training: ≥150,000 instances
Fine-tuning: ≥50,000 instances
Splits: 80% train, 10% validation, 10% test

Technical Requirements

Python 3.8+
PyTorch 2.0+
CUDA-capable GPU (8GB+ VRAM recommended)
Git (for cloning repositories)
GitHub account (optional but recommended for higher API rate limits)

🚀 Quick Start
1. Installation
bash# Clone the repository
git clone <your-repo-url>
cd if-statement-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy requests tqdm matplotlib editdistance
2. Get GitHub Token (Recommended)
Without a token, you're limited to 60 API requests/hour. With a token, you get 5000/hour.

Go to https://github.com/settings/tokens
Click "Generate new token (classic)"
Give it a name (e.g., "if-prediction-project")
Select scope: public_repo (read access to public repositories)
Generate and copy the token

3. Run Complete Pipeline
bash# Run entire pipeline with one command
python main.py --github-token YOUR_TOKEN_HERE

# Or with custom configuration
python main.py \
  --github-token YOUR_TOKEN_HERE \
  --num-repos 500 \
  --vocab-size 10000 \
  --pretrain-epochs 10 \
  --finetune-epochs 20 \
  --batch-size 32
4. Step-by-Step Execution
If you prefer to run steps individually:
bash# Step 1: Collect data from GitHub
python data_collection.py --token YOUR_TOKEN --repos 500 --min-stars 10

# Step 2: Train tokenizer
python tokenizer_training.py

# Step 3: Create datasets
python dataset_creation.py

# Step 4: Pre-train model
python training_script.py --mode pretrain --epochs 10

# Step 5: Fine-tune model
python training_script.py --mode finetune --epochs 20

# Step 6: Evaluate
python evaluation_script.py
📁 Project Structure
project/
├── data_collection.py          # GitHub API data collection
├── tokenizer_training.py       # Custom BPE tokenizer training
├── dataset_creation.py         # Pre-training & fine-tuning datasets
├── transformer_model.py        # Model architecture
├── training_script.py          # Training loops (pre-train & fine-tune)
├── evaluation_script.py        # Evaluation & inference
├── main.py                     # Complete pipeline orchestration
├── README.md                   # This file
│
├── python_repos/               # Cloned GitHub repositories
├── datasets/                   # Generated datasets
│   ├── pretraining_data.json
│   ├── finetuning_train.json
│   ├── finetuning_val.json
│   └── finetuning_test.json
│
├── checkpoints/                # Model checkpoints
│   ├── best_pretrain_model.pt
│   └── best_finetune_model.pt
│
└── results/                    # Evaluation results
    ├── evaluation_results.json
    ├── pretraining_curves.png
    └── finetuning_curves.png
🔧 Configuration Options
Data Collection

--num-repos: Number of repositories to process (default: 500)
--min-stars: Minimum stars for repository selection (default: 10)
--github-token: Your GitHub personal access token

Tokenizer

--vocab-size: Vocabulary size for BPE tokenizer (default: 10000)

Model Architecture

d_model: Hidden dimension (default: 512)
num_encoder_layers: Number of encoder layers (default: 6)
num_decoder_layers: Number of decoder layers (default: 6)
num_heads: Number of attention heads (default: 8)
d_ff: Feed-forward dimension (default: 2048)

Training

--pretrain-epochs: Pre-training epochs (default: 10)
--finetune-epochs: Fine-tuning epochs (default: 20)
--batch-size: Batch size (default: 32)
Learning rate: Auto-scheduled with warmup

📊 Expected Results
Data Collection

Target: 500 repositories
Expected functions: 100,000-200,000
Time: 2-4 hours (depending on network)

Pre-training

Dataset: 150,000 instances
Time: 8-12 hours on RTX 3070
Expected loss: ~2.5-3.0

Fine-tuning

Dataset: 50,000 instances (train)
Time: 4-6 hours on RTX 3070
Target metrics:

Exact Match: >30% (good), >50% (excellent)
Token Accuracy: >60%
Syntax Validity: >90%



🎓 Design Decisions Explained
1. Why GitHub API Instead of SEART?

Easier access: No registration required
Better control: Fine-grained search queries
Diversity: Topic-based search ensures varied code patterns
Rate limits: 5000 req/hour with token (sufficient)

2. Why Custom Tokenizer?

Project requirement: No pre-trained tokenizers allowed
Domain-specific: Learns Python code patterns
Efficiency: BPE balances vocabulary size and coverage
Task-specific tokens: Can define [IF_MASK] token

3. Why T5-style Architecture?

Seq2seq task: Naturally handles variable-length output
Proven architecture: Effective for code generation
Pre-training flexibility: Can use MLM on encoder
Decoder specialization: Learns to generate conditions

4. Why Include Full Function Context?

Better predictions: Understands variable types and scope
Realistic: Developers have full context when writing
Higher accuracy: Relationships between variables
Real-world applicability: More practical system

5. Why Two-Phase Training?

Pre-training: Learn general Python syntax and patterns
Fine-tuning: Specialize for if-condition prediction
Transfer learning: Leverages unlabeled code effectively
Better performance: Pre-training provides strong foundation

🐛 Troubleshooting
Out of Memory (OOM) Errors
bash# Reduce batch size
python main.py --batch-size 16

# Or reduce model size (edit transformer_model.py)
d_model = 256  # Instead of 512
num_layers = 4  # Instead of 6
Rate Limit Errors
bash# The script automatically handles rate limits
# But you can provide a token for higher limits
python main.py --github-token YOUR_TOKEN
Insufficient Data
bash# Increase repository count
python main.py --num-repos 1000

# Or lower minimum stars
python main.py --min-stars 5
CUDA Not Available
python# The code automatically falls back to CPU
# But training will be much slower (50-100x)
# Consider using Google Colab with free GPU
📈 Monitoring Training
Training Progress
The training scripts output real-time progress:
Pre-training: 100%|████████| 4688/4688 [1:23:45<00:00, loss=2.87, lr=0.0001]
Train Loss: 2.8732, Val Loss: 2.9145
✓ Checkpoint saved to best_pretrain_model.pt
Visualize Curves
Training and validation curves are automatically saved:
bash# View training curves
open pretraining_curves.png
open finetuning_curves.png
TensorBoard (Optional)
python# Add to training_script.py
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
writer.add_scalar('Loss/train', train_loss, epoch)
Then run: tensorboard --logdir=runs
🧪 Testing Your Model
Quick Test
pythonfrom evaluation_script import IfStatementPredictor
from tokenizer_training import PythonCodeTokenizer
from transformer_model import TransformerModel
import torch

# Load tokenizer
tokenizer = PythonCodeTokenizer()
tokenizer.load("python_tokenizer.pkl")

# Load model
model = TransformerModel(vocab_size=len(tokenizer.token_to_id))
checkpoint = torch.load("best_finetune_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Create predictor
predictor = IfStatementPredictor(model, tokenizer, device='cuda')

# Test prediction
test_code = """
def check_positive(x):
    if [IF_MASK]:
        return True
    return False
"""

prediction = predictor.predict(test_code)
print(f"Predicted: {prediction}")
# Expected: "x > 0" or similar
Evaluate on Custom Test Set
pythonfrom evaluation_script import evaluate_model

metrics, predictions = evaluate_model(
    model_path="best_finetune_model.pt",
    test_data_path="your_custom_test.json",
    tokenizer_path="python_tokenizer.pkl"
)
📝 Creating Your Own Test Cases
pythontest_cases = [
    {
        "input": "def is_even(n):\n    if [IF_MASK]:\n        return True\n    return False",
        "target": "n % 2 == 0"
    },
    {
        "input": "def max_val(a, b):\n    if [IF_MASK]:\n        return a\n    return b",
        "target": "a > b"
    }
]

import json
with open("my_test_cases.json", 'w') as f:
    json.dump(test_cases, f, indent=2)
🎯 Performance Tips
Improve Accuracy

More data: Collect from 1000+ repositories
Better filtering: Stricter quality criteria
Longer training: More epochs if not overfitting
Larger model: Increase d_model to 768
Data augmentation: Create variations of if conditions

Faster Training

Mixed precision: Use torch.cuda.amp
Gradient accumulation: Effective larger batch size
Multiple GPUs: Distributed training
Efficient data loading: Increase num_workers
Compile model: torch.compile() (PyTorch 2.0+)

Reduce Memory

Smaller batch size: 16 or 8
Gradient checkpointing: Trade compute for memory
Smaller model: Reduce layers/dimensions
Shorter sequences: Reduce max_len to 256

🔬 Advanced Usage
Resume Training
python# In training_script.py, add:
if Path("checkpoint.pt").exists():
    trainer.load_checkpoint("checkpoint.pt")
    start_epoch = checkpoint['epoch'] + 1
Custom Architecture
python# Modify transformer_model.py
model = TransformerModel(
    vocab_size=10000,
    d_model=768,  # Larger
    num_encoder_layers=8,  # Deeper
    num_decoder_layers=8,
    num_heads=12,
    d_ff=3072
)
Hyperparameter Tuning
pythonfor lr in [1e-4, 5e-5, 1e-5]:
    for batch_size in [16, 32, 64]:
        # Train with different configs
        # Track results in a table