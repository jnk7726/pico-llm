import argparse
import os
import glob
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tiktoken
from datasets import load_dataset
from collections import defaultdict
import time


import sys
import importlib.util


script_dir = os.path.dirname(os.path.abspath(__file__))
pico_llm_path = os.path.join(script_dir, "pico-llm.py")
spec = importlib.util.spec_from_file_location("pico_llm", pico_llm_path)
pico_llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pico_llm)


KGramMLPSeqModel = pico_llm.KGramMLPSeqModel
LSTMSeqModel = pico_llm.LSTMSeqModel
TransformerModel = pico_llm.TransformerModel
compute_next_token_loss = pico_llm.compute_next_token_loss
generate_text = pico_llm.generate_text
seq_collate_fn = pico_llm.seq_collate_fn
MixedSequenceDataset = pico_llm.MixedSequenceDataset
from utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained models")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Directory containing model checkpoints (e.g., runs/lstm_tinystories)")
    parser.add_argument("--test_size", type=int, default=1000,
                        help="Number of test samples to evaluate on. Default=1000.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation. Default=16.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on. Default=cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default=42.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file for metrics. Default=None (prints to stdout).")
    parser.add_argument("--generate_samples", action="store_true",
                        help="Generate text samples from each model for qualitative comparison.")
    parser.add_argument("--prompts", nargs="*", 
                        default=["Once upon a"],
                        help="Prompts for text generation. Default=['Once upon a']")
    parser.add_argument("--max_gen_tokens", type=int, default=50,
                        help="Maximum tokens to generate per prompt. Default=50.")
    
    # Model hyperparameters (with defaults matching training script)
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size. Default=50257 (GPT2).")
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Embedding size for LSTM and KGramMLP. Default=1024.")
    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length. Default=1024.")
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="K-gram window size. Default=3.")
    parser.add_argument("--kgram_num_layers", type=int, default=1,
                        help="Number of inner MLP layers for KGramMLP. Default=1.")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Transformer model dimension. Default=256.")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads. Default=4.")
    parser.add_argument("--n_blocks", type=int, default=4,
                        help="Number of transformer blocks. Default=4.")
    parser.add_argument("--mlp_hidden_dim", type=int, default=None,
                        help="MLP hidden dimension for transformer (default: 4 * d_model).")
    parser.add_argument("--eval_cooking_basics", action="store_true",
                        help="Evaluate on Cooking Basics dataset instead of TinyStories.")
    
    return parser.parse_args()


def find_model_files(run_dir):
    """Find all model checkpoint files in the run directory."""
    model_files = {}
    patterns = {
        "lstm_seq": "lstm_seq_final.pth",
        "kgram_mlp_seq": "kgram_mlp_seq_final.pth",
        "transformer_picollm": "transformer_picollm_final.pth"
    }
    
    for model_name, pattern in patterns.items():
        filepath = os.path.join(run_dir, pattern)
        if os.path.exists(filepath):
            model_files[model_name] = filepath
        else:
            # Try alternative patterns
            alt_pattern = os.path.join(run_dir, f"*{model_name}*.pth")
            matches = glob.glob(alt_pattern)
            if matches:
                model_files[model_name] = matches[0]
    
    return model_files


def load_model(model_name, model_path, args, device):
    """Load a model from checkpoint with appropriate architecture."""
    print(f"Loading {model_name} from {model_path}...")
    
    vocab_size = args.vocab_size
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    if model_name == "lstm_seq":
        model = LSTMSeqModel(
            vocab_size=vocab_size,
            embed_size=args.embed_size,
            hidden_size=args.embed_size
        )
    elif model_name == "kgram_mlp_seq":
        model = KGramMLPSeqModel(
            vocab_size=vocab_size,
            k=args.kgram_k,
            embed_size=args.embed_size,
            num_inner_layers=args.kgram_num_layers,
            chunk_size=1
        )
    elif model_name == "transformer_picollm":
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            max_seq_len=args.block_size,
            mlp_hidden_dim=args.mlp_hidden_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    model = torch.compile(model)
    # Load state dict
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"  ✓ Loaded {model_name} successfully")
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate a model on test data and return metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_tokens in dataloader:
            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)
            
            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)
            
            total_loss += loss.item()
            num_batches += 1
    
    if num_batches == 0:
        return {"loss": float('inf'), "perplexity": float('inf'), "num_batches": 0}
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "num_batches": num_batches
    }


def generate_samples_for_model(model, model_name, enc, prompts, max_tokens, device):
    """Generate text samples from a model for given prompts."""
    samples = {}
    model.eval()
    
    for prompt in prompts:
        try:
            text_greedy, _ = generate_text(
                model, enc, prompt, 
                max_new_tokens=max_tokens, 
                device=device,
                top_p=None
            )
            text_topp, _ = generate_text(
                model, enc, prompt,
                max_new_tokens=max_tokens,
                device=device,
                top_p=0.95
            )
            text_topp1, _ = generate_text(
                model, enc, prompt,
                max_new_tokens=max_tokens,
                device=device,
                top_p=1.00
            )
            samples[prompt] = {
                "greedy": text_greedy,
                "top_p_0.95": text_topp,
                "top_p_1.00": text_topp1
            }
        except Exception as e:
            print(f"  Warning: Error generating for prompt '{prompt}': {e}")
            samples[prompt] = {
                "greedy": f"ERROR: {str(e)}",
                "top_p_0.95": f"ERROR: {str(e)}"
            }
    
    return samples


def prepare_test_data(args):
    """Prepare test dataset from TinyStories or Cooking Knowledge Basics."""
    print("Loading test data...")
    enc = tiktoken.get_encoding("gpt2")
    
    if not args.eval_cooking_basics:
        # Load TinyStories test set
        dataset = load_dataset("roneneldan/TinyStories", split="validation")
        test_seqs = []
        for sample in test_dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:args.block_size]
            if len(tokens) > 0:
                test_seqs.append(tokens)
    else:
        # Load Cooking Basics dataset
        dataset = load_dataset("ktiyab/cooking-knowledge-basics")
        test_dataset = dataset['train'].select(range(5000, 5647))
        test_seqs = []
        for sample in test_dataset:
            text = "Question: " + sample['question'] + "\nAnswer: " + sample['response']
            tokens = enc.encode(text)
            tokens = tokens[:args.block_size]
            if len(tokens) > 0:
                test_seqs.append(tokens)
    
    
    
    print(f"Loaded {len(test_seqs)} test sequences")
    
    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return torch.tensor(self.sequences[idx], dtype=torch.long)
    
    test_dataset = TestDataset(test_seqs)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0, 
        collate_fn=seq_collate_fn,
        pin_memory=True if args.device.startswith("cuda") else False
    )
    
    return test_loader, enc


def main():
    args = parse_args()
    seed_everything(args.seed)
    
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    print(f"Run directory: {args.run_dir}")
    
    # Find model files
    model_files = find_model_files(args.run_dir)
    if not model_files:
        print(f"Error: No model files found in {args.run_dir}")
        print("Expected files: lstm_seq_final.pth, kgram_mlp_seq_final.pth, transformer_picollm_final.pth")
        return
    
    print(f"\nFound {len(model_files)} model(s): {list(model_files.keys())}")
    
    # Prepare test data
    test_loader, enc = prepare_test_data(args)
    
    # Evaluate each model
    results = {}
    all_samples = {}
    
    for model_name, model_path in model_files.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model = load_model(model_name, model_path, args, device)
        
        # Evaluate
        print("Running evaluation...")
        start_time = time.time()
        metrics = evaluate_model(model, test_loader, device)
        eval_time = time.time() - start_time
        metrics["eval_time_seconds"] = eval_time
        
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.4f}")
        print(f"  Evaluation time: {eval_time:.2f}s")
        
        results[model_name] = metrics
        
        # Generate samples if requested
        if args.generate_samples:
            print("Generating text samples...")
            samples = generate_samples_for_model(
                model, model_name, enc, args.prompts, 
                args.max_gen_tokens, device
            )
            all_samples[model_name] = samples
            
            for prompt, outputs in samples.items():
                print(f"\n  Prompt: '{prompt}'")
                print(f"    Greedy: {outputs['greedy']}")
                print(f"    Top-p 0.95:  {outputs['top_p_0.95']}")
                print(f"    Top-p 1.00:  {outputs['top_p_1.00']}")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Loss':<12} {'Perplexity':<12} {'Eval Time (s)':<15}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['loss']:<12.4f} {metrics['perplexity']:<12.2f} {metrics['eval_time_seconds']:<15.2f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['perplexity'])
    print(f"\nBest model (lowest perplexity): {best_model[0]} (PPL: {best_model[1]['perplexity']:.2f})")
    
    # Save results
    output_data = {
        "run_dir": args.run_dir,
        "test_size": args.test_size,
        "results": results,
        "samples": all_samples if args.generate_samples else None
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    else:
        print("\nFull results (JSON):")
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()



