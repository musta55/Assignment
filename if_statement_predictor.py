#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import pandas as pd
import parso
import random
import re
import torch
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
# Updated imports for the new tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForLanguageModeling,
)

# 1. CONFIGURATION

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
PRETRAIN_TARGET_SIZE = 150000   
FINETUNE_TARGET_SIZE = 50000    
NUM_REPOS_TO_FETCH = 20       
MIN_FUNCTION_TOKENS = 20      
TOKENIZER_VOCAB_SIZE = 20000
MAX_TRAIN_EPOCHS_PRETRAIN = 3
MAX_TRAIN_EPOCHS_FINETUNE = 5
BATCH_SIZE = 1
GRAD_ACCUM = 32
MAX_SEQ_LENGTH = 256
MAX_TARGET_LENGTH = 128
EVAL_STEPS = 2000
SAVE_STEPS = 2000

MASK_TOKEN = "<extra_id_0>"

DATA_ROOT = Path("./if_statement_project")
RAW_CODE_DIR = DATA_ROOT / "raw_code"
TOKENIZER_PATH = DATA_ROOT / "python_tokenizer.json"
ALL_FUNCTIONS_FILE = DATA_ROOT / "all_functions.txt"
PRETRAIN_FILE = DATA_ROOT / "pretrain.txt"
PRETRAIN_VALID_FILE = DATA_ROOT / "pretrain_valid.txt"
FINETUNE_TRAIN_FILE = DATA_ROOT / "finetune_train.txt"
FINETUNE_VALID_FILE = DATA_ROOT / "finetune_valid.txt"
FINETUNE_TEST_FILE = DATA_ROOT / "finetune_test.txt"
PRETRAINED_MODEL_DIR = DATA_ROOT / "pretrained_t5"
FINETUNED_MODEL_DIR = DATA_ROOT / "finetuned_if_model"
PROVIDED_TESTSET_PATH = DATA_ROOT / "provided_testset_to_process.csv"

DATA_ROOT.mkdir(exist_ok=True)
RAW_CODE_DIR.mkdir(exist_ok=True)

# 2. DATA COLLECTION & PROCESSING

def get_popular_python_repos(n=NUM_REPOS_TO_FETCH):
    """Fetches the most starred Python repositories from GitHub."""
    print(f"Fetching {n} popular Python repository names...")
    headers = {"Authorization": f"Bearer {GITHUB_API_KEY}"}
    params = {"q": "language:Python", "sort": "stars", "order": "desc", "per_page": n}
    resp = requests.get("https://api.github.com/search/repositories", headers=headers, params=params)
    resp.raise_for_status()
    return [repo['full_name'] for repo in resp.json()['items']]

def get_python_files_from_repo(repo_full_name):
    """Recursively fetches all .py file paths from a repository."""
    headers = {"Authorization": f"Bearer {GITHUB_API_KEY}"}
    api_url = f"https://api.github.com/repos/{repo_full_name}/git/trees/main?recursive=1"
    resp = requests.get(api_url, headers=headers)
    if resp.status_code != 200:
        return []
    tree = resp.json().get('tree', [])
    py_files = [item['path'] for item in tree if item['path'].endswith('.py')]
    return py_files

def download_and_save_code(repo_full_name, file_path):
    """Downloads a single file's content and saves it locally."""
    raw_url = f"https://raw.githubusercontent.com/{repo_full_name}/main/{file_path}"
    try:
        resp = requests.get(raw_url, timeout=10)
        if resp.status_code == 200:
            safe_filename = file_path.replace('/', '_')
            with open(RAW_CODE_DIR / f"{repo_full_name.replace('/', '_')}_{safe_filename}", "w", encoding="utf-8") as f:
                f.write(resp.text)
            return True
    except (requests.exceptions.RequestException, Exception):
        return False
    return False

def extract_functions_from_code(code_str):
    """Uses parso to extract function bodies from a string of Python code."""
    try:
        module = parso.parse(code_str)
        return [
            node.get_code()
            for node in module.iter_funcdefs()
            if len(node.get_code().split()) > MIN_FUNCTION_TOKENS
        ]
    except Exception:
        return []

# --- 3. TOKENIZER TRAINING ---

def train_tokenizer(file_path):
    """Trains a Hugging Face BPE tokenizer from the collected functions."""
    print("Training Hugging Face BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", MASK_TOKEN]
    )

    tokenizer.train(files=[str(file_path)], trainer=trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    print("Tokenizer training complete.")

# --- 4. DATASET CREATION ---

def create_pretrain_dataset(functions, tokenizer):
    """Creates MLM dataset for T5 span corruption."""
    print(f"Creating pre-training dataset with {PRETRAIN_TARGET_SIZE} instances...")
    dataset = []
    pbar = tqdm(total=PRETRAIN_TARGET_SIZE)
    
    attempts = 0
    max_attempts = PRETRAIN_TARGET_SIZE * 10  # Prevent infinite loop
    
    while len(dataset) < PRETRAIN_TARGET_SIZE and attempts < max_attempts:
        attempts += 1
        func = random.choice(functions)
        
        tokens = tokenizer.encode(func).tokens
        if len(tokens) < 10:
            continue

        num_to_mask = max(1, int(len(tokens) * 0.15))
        
        masked_indices = sorted(random.sample(range(len(tokens)), k=min(num_to_mask, len(tokens))))
        
        input_parts = []
        target_parts = []
        
        last_end = 0
        mask_counter = 0
        for i in masked_indices:
            input_parts.extend(tokens[last_end:i])
            input_parts.append(f"<extra_id_{mask_counter}>")
            target_parts.append(f"<extra_id_{mask_counter}>")
            target_parts.append(tokens[i])
            mask_counter += 1
            last_end = i + 1
        
        input_parts.extend(tokens[last_end:])
        target_parts.append(f"<extra_id_{mask_counter}>")

        input_str = " ".join(input_parts)
        target_str = " ".join(target_parts)
        
        dataset.append(f"{input_str}\t{target_str}")
        pbar.update(1)
        
    pbar.close()
    
    if len(dataset) < PRETRAIN_TARGET_SIZE:
        print(f"Warning: Only created {len(dataset)} examples instead of {PRETRAIN_TARGET_SIZE}")
    
    # Split dataset into train (90%) and validation (10%) for pre-training
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    valid_data = dataset[split_idx:]
    
    # Save train and validation separately
    with open(PRETRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
    
    pretrain_valid_file = DATA_ROOT / "pretrain_valid.txt"
    with open(pretrain_valid_file, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_data))
    
    print(f"Pre-training: {len(train_data)} train, {len(valid_data)} validation samples")

def create_finetune_dataset(functions):
    """Creates the if-statement masking dataset."""
    print(f"Creating fine-tuning dataset with {FINETUNE_TARGET_SIZE} instances...")
    dataset = []
    pbar = tqdm(total=FINETUNE_TARGET_SIZE)
    
    if_regex = re.compile(r'if\s+(.+?):')
    
    random.shuffle(functions)
    for func in functions:
        matches = list(if_regex.finditer(func))
        if not matches:
            continue
            
        match = random.choice(matches)
        condition = match.group(1).strip()
        
        masked_func = func[:match.start(1)] + MASK_TOKEN + func[match.end(1):]
        
        dataset.append(f"{masked_func}\t{condition}")
        pbar.update(1)
        if len(dataset) >= FINETUNE_TARGET_SIZE:
            break
    
    pbar.close()
    
    if len(dataset) < FINETUNE_TARGET_SIZE:
        print(f"Warning: Only created {len(dataset)} examples instead of {FINETUNE_TARGET_SIZE}")
    
    # Split and save
    train_end = int(len(dataset) * 0.8)
    valid_end = int(len(dataset) * 0.9)
    
    with open(FINETUNE_TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset[:train_end]))
    with open(FINETUNE_VALID_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset[train_end:valid_end]))
    with open(FINETUNE_TEST_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset[valid_end:]))

# --- 5. MODEL TRAINING ---

def run_training(is_pretraining):
    """A general function to run either pre-training or fine-tuning."""
    
    model_dir = PRETRAINED_MODEL_DIR if is_pretraining else FINETUNED_MODEL_DIR
    train_file = PRETRAIN_FILE if is_pretraining else FINETUNE_TRAIN_FILE
    valid_file = PRETRAIN_FILE if is_pretraining else FINETUNE_VALID_FILE
    
    print(f"\n--- Starting {'Pre-training' if is_pretraining else 'Fine-tuning'} ---")

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load custom tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    tokenizer.pad_token = "<pad>"
    
    # Load or initialize model
    if is_pretraining:
        config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model= 256,  
            d_ff= 1024,
            num_layers= 4,   
            num_heads= 4,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        model = T5ForConditionalGeneration(config)
        print(f"Initialized model with {model.num_parameters():,} parameters")
    else:
        model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_DIR)

    # Load and process dataset
    dataset = load_dataset('text', data_files={'train': str(train_file), 'validation': str(valid_file)})
    
    def tokenize_function(examples):
        source, target = [], []
        for line in examples['text']:
            if '\t' in line:
                s, t = line.split('\t', 1)
                source.append(s)
                target.append(t)
        
        if not source:  # Handle empty batches
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        prefix = "complete the python code: "
        source = [prefix + s for s in source]

        model_inputs = tokenizer(
            source, 
            max_length=MAX_SEQ_LENGTH, 
            padding="max_length", 
            truncation=True
        )
        labels = tokenizer(
            target, 
            max_length=MAX_TARGET_LENGTH, 
            padding="max_length", 
            truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text'],
        desc="Tokenizing"
    )

    # Training arguments
    num_epochs = MAX_TRAIN_EPOCHS_PRETRAIN if is_pretraining else MAX_TRAIN_EPOCHS_FINETUNE
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=num_epochs,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        learning_rate=5e-4 if is_pretraining else 1e-4,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        logging_steps= 500,
        fp16=True,  # Use mixed precision for faster training
        dataloader_num_workers=0,
        report_to="none",  # Disable wandb/tensorboard
        max_grad_norm=1.0,  # Gradient clipping
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train with error handling
    try:
        print(f"Starting training for {num_epochs} epoch(s)...")
        print(f"Training samples: {len(tokenized_datasets['train'])}")
        print(f"Validation samples: {len(tokenized_datasets['validation'])}")
        trainer.train()
        trainer.save_model()
        print(f"--- {'Pre-training' if is_pretraining else 'Fine-tuning'} Complete ---")
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise

# --- 7. MAIN EXECUTION (Steps 1-5) ---
if __name__ == '__main__':
    
    print(f"\nConfiguration:")
    print(f"  Pretrain samples: {PRETRAIN_TARGET_SIZE}")
    print(f"  Finetune samples: {FINETUNE_TARGET_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRAD_ACCUM}")
    print(f"  Pretrain epochs: {MAX_TRAIN_EPOCHS_PRETRAIN}")
    print(f"  Finetune epochs: {MAX_TRAIN_EPOCHS_FINETUNE}\n")
    
    # --- Step 1: Collect and Process Data ---
    if not list(RAW_CODE_DIR.glob("*.py")):
        repo_names = get_popular_python_repos()
        for repo_name in tqdm(repo_names, desc="Downloading Repos"):
            py_files = get_python_files_from_repo(repo_name)
            for file_path in tqdm(py_files, desc=f"Files in {repo_name}", leave=False):
                download_and_save_code(repo_name, file_path)
    
    if not ALL_FUNCTIONS_FILE.exists():
        all_funcs = []
        for code_file in tqdm(list(RAW_CODE_DIR.glob("*.py")), desc="Extracting Functions"):
            with open(code_file, 'r', encoding='utf-8') as f:
                all_funcs.extend(extract_functions_from_code(f.read()))
        
        all_funcs = list(set(all_funcs))  # Deduplicate
        with open(ALL_FUNCTIONS_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(all_funcs))
        print(f"Extracted and saved {len(all_funcs)} unique functions.")
    else:
        with open(ALL_FUNCTIONS_FILE, "r", encoding="utf-8") as f:
            all_funcs = f.read().splitlines()
        print(f"Loaded {len(all_funcs)} functions from cache.")

    # --- Step 2: Train Tokenizer ---
    if not TOKENIZER_PATH.exists():
        train_tokenizer(ALL_FUNCTIONS_FILE)

    # --- Step 3: Create Datasets ---
    if not PRETRAIN_FILE.exists() or not FINETUNE_TRAIN_FILE.exists():
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        create_pretrain_dataset(all_funcs, tokenizer)
        create_finetune_dataset(all_funcs)
    
    # --- Step 4: Pre-train the Model ---
    if not (PRETRAINED_MODEL_DIR / "pytorch_model.bin").exists():
        run_training(is_pretraining=True)

    # --- Step 5: Fine-tune the Model ---
    if not (FINETUNED_MODEL_DIR / "pytorch_model.bin").exists():
        run_training(is_pretraining=False)
    
    print("\n--- Steps 1-5 Complete. Run next cell for evaluation. ---")


# In[ ]:


# --- 6. EVALUATION ---

def load_provided_test_set(file_path):
    """Loads and parses the provided test CSV with specific format."""
    df = pd.read_csv(file_path)
    return df['code'].tolist()


def evaluate_and_create_csv(model, tokenizer, test_data, output_filename):
    """Runs model predictions and saves results to a CSV file."""
    print(f"Evaluating and creating {output_filename}...")

    results = []
    device = "cuda"
    model.to(device)
    if_regex = re.compile(r'if\s+(.+?):')

    for code in tqdm(test_data, desc=f"Predicting for {output_filename}"):
        matches = list(if_regex.finditer(code))
        if not matches:
            continue

        match = matches[0]
        expected_condition = match.group(1).strip()

        input_code = code[:match.start(1)] + MASK_TOKEN + code[match.end(1):]
        prefix = "complete the python code: "

        # FIX: Stop returning token_type_ids which T5 doesn't support
        inputs = tokenizer(
            prefix + input_code,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        ).to(device)

        # Only forward valid fields
        input_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        with torch.no_grad():
            output = model.generate(
                **input_kwargs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        predicted_condition = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        seq_log_prob = output.sequences_scores[0].item()
        score = min(100.0, max(0.0, torch.exp(torch.tensor(seq_log_prob)).item() * 100))

        results.append({
            "Input provided to the model": input_code,
            "Whether the prediction is correct": predicted_condition.strip() == expected_condition,
            "Expected if condition": expected_condition,
            "Predicted if condition": predicted_condition,
            "Prediction score (0-100)": round(score, 2)
        })

    if not results:
        print(f"WARNING: No valid test cases found for {output_filename}")
        return

    pd.DataFrame(results).to_csv(output_filename, index=False)
    print(f"Successfully saved {len(results)} results to {output_filename}")

    correct = sum(1 for r in results if r["Whether the prediction is correct"])
    accuracy = (correct / len(results)) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(results)})")


# --- 7. MAIN EXECUTION ---
if __name__ == '__main__':
    print("\n--- Starting Final Evaluation ---")
    final_model = T5ForConditionalGeneration.from_pretrained(FINETUNED_MODEL_DIR)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    tokenizer.pad_token = "<pad>"

    # Evaluate generated fine-tune test set
    with open(FINETUNE_TEST_FILE, "r", encoding="utf-8") as f:
        generated_test_data = [line.split('\t')[0] for line in f.read().splitlines()]
    evaluate_and_create_csv(final_model, tokenizer, generated_test_data, "generated-testset.csv")

    # Evaluate user-provided test set
    if PROVIDED_TESTSET_PATH.exists():
        provided_test_data = load_provided_test_set(PROVIDED_TESTSET_PATH)
        evaluate_and_create_csv(final_model, tokenizer, provided_test_data, "provided-testset.csv")
    else:
        print(f"Warning: Provided test set missing at '{PROVIDED_TESTSET_PATH}' — skipping evaluation.")

    print("\n--- PIPELINE COMPLETE ---")

