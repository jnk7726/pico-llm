import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from utils import seed_everything

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. Default=42.")

    ## Weights & Biases project args                   
    parser.add_argument("--wandb_project", type=str, default="pico-llm",
                        help="If set, the Weights & Biases project name to log training metrics to. Default=None (no logging).")
    parser.add_argument("--use_wandb", action="store_true",
                        help="If set, enable Weights & Biases logging.")
    parser.add_argument("--wandb_run_name", type=str, default="lstm_seq_model",
                        help="If set, the Weights & Biases run name. Default=None (automatic)."
                        )

    parser.add_argument("--log_interval_steps", type=int, default=100,
                        help="Log training metrics every N steps. Default=100.")
    parser.add_argument("--sample_interval_seconds", type=int, default=30,
                        help="Sample model outputs every N seconds. Default=30.")

    ## Training hyperparameters
    parser.add_argument("--num_epochs", type=int,
                        help="Number of training epochs to run.",
                        default=3)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training. Default=16.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer. Default=1e-3.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for data loading. Default=4.")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default
    parser.add_argument("--model_type", type=str, default="lstm_seq",
                        help="Type of model to train: kgram_mlp_seq, lstm_seq, transformer_picollm. Default=lstm_seq",
                        choices=["kgram_mlp_seq", "lstm_seq", "transformer_picollm"])

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")
    parser.add_argument("--use_compile", action="store_true",
                        help="If set, use torch.compile() to potentially speed up the model.")
    
    # Transformer-specific arguments
    parser.add_argument("--d_model", type=int, default=512,
                        help="Dimension of transformer model embeddings. Default=512.")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads per transformer block. Default=8.")
    parser.add_argument("--n_blocks", type=int, default=6,
                        help="Number of transformer blocks. Default=6.")
    parser.add_argument("--mlp_hidden_dim", type=int, default=None,
                        help="Hidden dimension for MLP in transformer (default: 4 * d_model).")

    args = parser.parse_args()
    return args


################################################################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1, activation=nn.ReLU()):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        init_layer = nn.Linear(k * embed_size, embed_size)
        layers = [init_layer, activation]
        inner_layer = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            activation
        )
        inner_layers = nn.ModuleList([inner_layer for _ in range(num_inner_layers)])
        self.net = nn.Sequential(
            *layers,
            *inner_layers,
            nn.Linear(embed_size, vocab_size)
        )

    def forward(self, tokens_seq):
        """
        Fully vectorized k-gram MLP forward pass using unfold.
        tokens_seq: (seq_len, batch)
        Returns: (seq_len, batch, vocab_size)
        """
        seq_len, batch_size = tokens_seq.shape
        k = self.k

        # Pad the sequence at the beginning with zeros for k-gram context
        pad = torch.zeros(k-1, batch_size, dtype=tokens_seq.dtype, device=tokens_seq.device)
        padded_seq = torch.cat([pad, tokens_seq], dim=0)  # (seq_len + k - 1, batch)

        # Transpose to (batch, seq_len + k - 1)
        padded_seq = padded_seq.transpose(0, 1)
        # Unfold to get k-grams: (batch, seq_len, k)
        contexts = padded_seq.unfold(1, k, 1)
        # Transpose to (seq_len, batch, k)
        contexts = contexts.transpose(0, 1)  # (seq_len, batch, k)

        # Get embeddings for all context tokens
        context_embeds = self.embedding(contexts)  # (seq_len, batch, k, embed_size)
        context_flat = context_embeds.reshape(seq_len, batch_size, k * self.embed_size)
        logits = self.net(context_flat)
        return logits


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Causal Decoder-Only Transformer
#    Following Llama3 architecture: RMSNorm, SiLU activation, residual connections
################################################################################

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    RMSNorm(x) = (x / RMS(x)) * gamma
    RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        """
        x: (..., dim) - any shape ending in dim
        Returns: same shape as x
        """
        # Compute RMS for each element in dim, input: (seq_len, batch, d_model)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # output: (seq_len, batch, 1)
        
        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed


class CausalSelfAttention(nn.Module):
    """
    Single causal self-attention head for scaled dot-product attention
    """
    def __init__(self, d_model, head_dim):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(head_dim, d_model, bias=False)
        
        # Scale factor for attention scores
        self.scale = head_dim ** -0.5
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        Returns: (seq_len, batch, d_model)
        """
        seq_len, batch_size, d_model = x.shape
        
        # Project to Q, K, V
        # all shapes: (seq_len, batch, head_dim)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # attention scores: Q @ K^T / sqrt(head_dim)
        # transpose for batch matrix multiplication
        # all shapes: (batch, seq_len, head_dim)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        # attention scores: (batch, seq_len, seq_len)
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        
        # causal mask to prevent attending to future positions
        # lower triangular mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.view(1, seq_len, seq_len)
        
        # applying mask by setting future positions to -inf to avoid summation to 1 issues
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # handling potential NaN from softmax of all -inf (shouldn't happen with proper input)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # applying attention to values
        attn_output = torch.bmm(attn_weights, V)  # (batch, seq_len, head_dim)
        attn_output = attn_output.transpose(0, 1)  # (seq_len, batch, head_dim)
        
        # projecting back to d_model
        output = self.out_proj(attn_output)  # (seq_len, batch, d_model)
        
        return output


class MLP(nn.Module):
    """
    Feed-forward network (MLP) for the transformer blocks.
    Architecture: Linear -> SiLU -> Linear
    Usual hidden dimension is 4 * d_model (source: GPT-2 and Llama)
    """
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * d_model
        
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        Returns: (seq_len, batch, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with:
    1. Multi-head attention (where heads are summed together)
    2. MLP
    3. Residual connections (skip connections)
    4. RMSNorm before each sub-layer (like Llama)
    
    Architecture:
    x -> RMSNorm -> [sum of attention heads] -> residual -> 
         RMSNorm -> MLP -> residual -> output
    """
    def __init__(self, d_model, n_heads, mlp_hidden_dim=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # calculating head dimension
        self.head_dim = d_model // n_heads
        
        # pre-norm for attention
        self.attn_norm = RMSNorm(d_model)
        
        # creating multiple attention heads
        self.attention_heads = nn.ModuleList([
            CausalSelfAttention(d_model, self.head_dim)
            for _ in range(n_heads)
        ])
        
        # pre-norm for MLP
        self.mlp_norm = RMSNorm(d_model)
        
        # MLP
        self.mlp = MLP(d_model, mlp_hidden_dim)
    
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        Returns: (seq_len, batch, d_model)
        """

        normed_x = self.attn_norm(x)
        
        # summing outputs from all attention heads
        attn_output = sum(head(normed_x) for head in self.attention_heads)
        
        # residual connection
        x = x + attn_output
        
        # MLP block with residual connection
        normed_x = self.mlp_norm(x)
        mlp_output = self.mlp(normed_x)
        
        # residual connection
        x = x + mlp_output
        
        return x


class TransformerModel(nn.Module):
    """
    Causal decoder-only transformer model.
    Architecture:
    - Token embeddings
    - Positional embeddings (learned)
    - N transformer blocks
    - Final RMSNorm
    - Output projection to vocabulary
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Dimension of model embeddings
        n_heads: Number of attention heads per block
        n_blocks: Number of transformer blocks
        max_seq_len: Maximum sequence length for positional embeddings
        mlp_hidden_dim: Hidden dimension for MLP (default: 4 * d_model)
    """
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=8, n_blocks=6, 
                 max_seq_len=2048, mlp_hidden_dim=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.max_seq_len = max_seq_len
        
        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # positional embedding (learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_hidden_dim)
            for _ in range(n_blocks)
        ])
        
        # final normalization
        self.final_norm = RMSNorm(d_model)
        
        # output projection to vocabulary (language modeling head, unembedding layer)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # weight tying to share weights between token embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight
        
        # initializing weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initializing weights with normal distribution
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens_seq):
        """
        Forward pass through the transformer.
        
        Args:
            tokens_seq: (seq_len, batch) - input token indices
        
        Returns:
            logits: (seq_len, batch, vocab_size) - output logits for next token prediction
        """
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device
        
        # checking sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence Length {seq_len} Exceeds Maximum {self.max_seq_len}")
        
        # getting token embeddings
        token_emb = self.token_embedding(tokens_seq)  # (seq_len, batch, d_model)
        
        # getting positional embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(1).expand(seq_len, batch_size)  # (seq_len, batch)
        pos_emb = self.position_embedding(positions)  # (seq_len, batch, d_model)
        
        # combining embeddings
        x = token_emb + pos_emb  # (seq_len, batch, d_model)
        
        # applying transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # final normalization
        x = self.final_norm(x)
        
        # projecting to vocabulary
        logits = self.lm_head(x)  # (seq_len, batch, vocab_size)
        
        return logits

################################################################################
# 6. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    """
    logits: 1D tensor of shape (vocab_size,)
    p: cumulative probability threshold
    Returns: sampled token index
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff
    cutoff = cumulative_probs > p
    if torch.any(cutoff):
        cutoff_idx = torch.nonzero(cutoff)[0].item() + 1
    else:
        cutoff_idx = len(sorted_probs)

    # Only keep tokens within the nucleus
    nucleus_probs = sorted_probs[:cutoff_idx]
    nucleus_indices = sorted_indices[:cutoff_idx]

    # Normalize
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    # Sample
    sampled_idx = torch.multinomial(nucleus_probs, 1).item()
    return nucleus_indices[sampled_idx].item()


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
            next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 7. Training
################################################################################

def init_wandb(args, project_name: str, run_name: str = None):
    """Initialize Weights & Biases logging."""
    wandb.init(project=project_name, name=run_name, config=args)


def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    use_wandb=False):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "step": global_step,
                        "partial_avg_loss": avg_part_loss
                    })
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")


################################################################################
# 8. Main
################################################################################

def main():
    args = parse_args()
    seed_everything(args.seed)
    # Additional local variables from arguments

    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = args.log_interval_steps
    sample_interval_seconds = args.sample_interval_seconds

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    if args.use_wandb:
        init_wandb(args, project_name=args.wandb_project, run_name=args.wandb_run_name)
    
    ############################################################################
    # Data
    ############################################################################

    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        max_seq_len=block_size,
        mlp_hidden_dim=args.mlp_hidden_dim
    ).to(device)

    if args.model_type == "kgram_mlp_seq":
        models = {
            "kgram_mlp_seq": kgram_model if not args.use_compile else torch.compile(kgram_model),
        }
    elif args.model_type == "lstm_seq":
        models = {
            "lstm_seq": lstm_model if not args.use_compile else torch.compile(lstm_model),
        }
    elif args.model_type == "transformer_picollm":
        models = {
            "transformer_picollm": transformer if not args.use_compile else torch.compile(transformer),
        }
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Do not support multiple model training in one run for simplicity
    if len(models) > 1:
        raise ValueError("Multiple model training is not supported.")

    ############################################################################
    # Train each model
    ############################################################################

    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=None, #TODO
            prompt=args.prompt,
            use_wandb=args.use_wandb
        )

        # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")



if __name__ == "__main__":
    main()

