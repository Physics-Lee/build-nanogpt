import os
import math
import time
import inspect # for introspection
from dataclasses import dataclass # for configuration
import torch # pytorch
import torch.nn as nn # neural network modules
from torch.nn import functional as F # for activation functions etc.
from hellaswag import render_example, iterate_examples # for HellaSwag eval
import numpy as np
# -----------------------------------------------------------------------------

if torch.cuda.is_available():
    print("PyTorch is using GPU.")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is using CPU.")
    print(f"Device name: {torch.get_device_name(0)}")
# Causal Self-Attention
class CausalSelfAttention(nn.Module): # causal self-attention

    # Causal Self-Attention
    def __init__(self, config): # initialize causal self-attention
        super().__init__() # initialize parent class
        assert config.n_embd % config.n_head == 0 # ensure divisibility
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 3* for q, k, v
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # project back to n_embd
        self.c_proj.NANOGPT_SCALE_INIT = 1 # special init for residual branch
        # regularization
        self.n_head = config.n_head # number of heads
        self.n_embd = config.n_embd # embedding dimension

    # Forward pass
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module): # feedforward network

    def __init__(self, config): # initialize feedforward network
        super().__init__() # initialize parent class
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) # 4* expansion
        self.gelu    = nn.GELU(approximate='tanh') # GELU non-linearity
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) # project back to n_embd
        self.c_proj.NANOGPT_SCALE_INIT = 1 # special init for residual branch

    def forward(self, x):
        x = self.c_fc(x) # (B, T, 4 * n_embd)
        x = self.gelu(x) # (B, T, 4 * n_embd)
        x = self.c_proj(x) # (B, T, n_embd)
        return x

class Block(nn.Module): # transformer block

    def __init__(self, config): # initialize transformer block
        super().__init__() # initialize parent class
        self.ln_1 = nn.LayerNorm(config.n_embd) # layer norm 1
        self.attn = CausalSelfAttention(config) # causal self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd) # layer norm 2
        self.mlp = MLP(config) # feedforward network

    def forward(self, x): # forward pass
        x = x + self.attn(self.ln_1(x)) # residual connection 1
        x = x + self.mlp(self.ln_2(x)) # residual connection 2
        return x

@dataclass # data class for configuration
class GPTConfig: # configuration for GPT model
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module): # GPT model

    def __init__(self, config): # initialize model
        super().__init__() # initialize parent class
        self.config = config # save config

        # build transformer
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # final layer norm
        ))

        # language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # tie weights

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module): # initialize weights
        if isinstance(module, nn.Linear): # linear layer
            print(f"Initializing Linear layer: {module}")
            std = 0.02 # standard deviation
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # special init for residual branch
                std *= (2 * self.config.n_layer) ** -0.5 # rescale stddev
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # normal init
            if module.bias is not None: # if bias exists
                torch.nn.init.zeros_(module.bias) # zero init
        elif isinstance(module, nn.Embedding): # embedding layer
            print(f"Initializing Embedding layer: {module}")
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # normal init
        elif isinstance(module, nn.LayerNorm): # layer normalization
            print(f"Initializing LayerNorm layer: {module}")
            torch.nn.init.zeros_(module.bias) # bias to zero
            torch.nn.init.ones_(module.weight) # weight to one

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # idx.device stands for the device of idx, i.e. cpu or cuda
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod # class method to load pretrained model
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # supported types, paras: 124M, 350M, 774M, 1558M
        from transformers import GPT2LMHeadModel # import from transformers
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args) # create config
        model = GPT(config) # create model
        sd = model.state_dict() # get model state dict
        sd_keys = sd.keys() # get state dict keys
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # load pretrained model
        sd_hf = model_hf.state_dict() # get state dict

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys() # get hf state dict keys
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}" # ensure same number of keys
        for k in sd_keys_hf: # iterate over hf keys
            if any(k.endswith(w) for w in transposed): # special treatment for transposed weights
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape # ensure shapes match when transposed
                with torch.no_grad(): # no grad context
                    sd[k].copy_(sd_hf[k].t()) # copy with transpose
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ------------------------------------------------------------------------------

import tiktoken # for tokenization

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B # batch size
        self.T = T # sequence length

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2") # get GPT-2 tokenizer
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"dataset length in characters: {len(text)}")
        print(f"dataset length in tokens: {len(self.tokens)}")
        print(f"vocab size: {enc.n_vocab}")
        
        # print batch numbers
        self.n = len(self.tokens) // (B * T)
        print(f"data loader will produce {self.n} batches of size ({B}, {T})")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1] # (B * T + 1,)
        x = buf[:-1].view(B, T) # (B, T) 
        y = buf[1:].view(B, T) # (B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y
    
# ------------------------------------------------------------------------------

# load
train_loader = DataLoaderLite(B=16, T=1024) # create a data loader

# torch.set_float32_matmul_precision('high') # use high precision for matmul

#
enc = tiktoken.get_encoding("gpt2") # get GPT-2 tokenizer

# create the model
# model = GPT.from_pretrained('gpt2') # load pretrained model
model = GPT(GPTConfig(vocab_size = 50304)) # load a fresh model
print("number of parameters: %.2fM" % (sum(p.numel() for p in model.parameters())/1e6,))
model.to('cuda') # move model to GPU
# model = torch.compile(model) # compile the model (optional, requires PyTorch 2.0+) and LINUX

# get lr
def get_lr(it): # learning rate schedule
    # cosine learning rate decay with linear warmup
    # 1) linear warmup for warmup_iters steps
    # 2) cosine decay down to min_lr over total_iters steps
    warmup_iters = 1000
    total_iters = 20000
    min_lr = 1e-5
    max_lr = 1e-3
    if it < warmup_iters:
        return max_lr * (it+1) / warmup_iters
    if it > total_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (total_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay
    return min_lr + coeff * (max_lr - min_lr)

# def configure_optimizers(self, weight_decay, learning_rate, device_type):
#     # start with all of the candidate parameters (that require grad)
#     param_dict = {pn: p for pn, p in self.named_parameters()}
#     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
#     # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
#     # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
#     decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
#     nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
#     optim_groups = [
#         {'params': decay_params, 'weight_decay': weight_decay},
#         {'params': nodecay_params, 'weight_decay': 0.0}
#     ]
#     num_decay_params = sum(p.numel() for p in decay_params)
#     num_nodecay_params = sum(p.numel() for p in nodecay_params)
#     if master_process:
#         print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
#         print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
#     # Create AdamW optimizer and use the fused version if it is available
#     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#     use_fused = fused_available and device_type == "cuda"
#     if master_process:
#         print(f"using fused AdamW: {use_fused}")
#     optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
#     return optimizer


# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas = (0.9, 0.95), eps=1e-8)
# optimizer = configure_optimizers(model, weight_decay=0.1, learning_rate=1e-3, device_type='cuda')

for i in range(50): # 10 iterations of training
    t0 = time.time()
    x, y = train_loader.next_batch() # get batch
    x, y = x.to('cuda'), y.to('cuda') # move to GPU
    optimizer.zero_grad(set_to_none=True) # reset gradients
    with torch.autocast(device_type='cuda', dtype=torch.float16): # use autocast for mixed precision
        logits, loss = model(x, y) # forward pass
        # import code; code.interact(local=locals()) # for debugging
    loss.backward() # backward pass
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
    lr = get_lr(i) # get learning rate 
    for param_group in optimizer.param_groups: # 
        param_group['lr'] = lr # set learning rate
    optimizer.step() # update parameters
    torch.cuda.synchronize() # <--- 关键！强制CPU等待GPU完成所有工作
    t1 = time.time()
    # 将所有指标合并到一行打印
    print(f"iteration {i+1} | loss: {loss.item():.4f} | lr: {lr:.4e} | norm: {norm.item():.4f} | time: {t1 - t0:.2f}s | tokens/sec: {(x.numel()) / (t1 - t0):.0f}")

# toy model
# tokens = enc.encode("Explain the moon landing to a 6 year old in a few sentences.") # encode input text
# tokens = torch.tensor(tokens, dtype=torch.long) # (7,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 7)  (batch size, sequence length)
# x = tokens.to('cuda') # move tokens to GPU

# gernerate text
# torch.manual_seed(42) # for reproducibility
# torch.cuda.manual_seed(42) # for reproducibility
# with torch.no_grad(): # no grad context
#     for _ in range(max_length):
#         logits, _ = model(x.to('cuda')) # (5, T, 50257)
#         logits = logits[:, -1, :] # (5, 50257) take the last token
#         probs = F.softmax(logits, dim=-1) # (5, 50257) softmax to get probabilities
#         topk_probs, topk_indices = torch.topk(probs, k=10, dim=-1) # (5, 10) get top 10 probabilities and indices
#         ix = torch.multinomial(topk_probs, num_samples=1) # (5, 1) sample from the top 10
#         next_token = torch.gather(topk_indices, -1, ix) # (5, 1) get the actual token
#         x = torch.cat((x, next_token), dim=1) # (5, T+1) append to the sequence

# # print generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length + 7].tolist() # get tokens
#     text = enc.decode(tokens) # decode tokens to text
#     print(">", text) # print generated text

# print generated text
# for i in range(num_return_sequences):
#     generated_tokens = x[i].cpu().numpy() # move to cpu and convert to numpy
#     generated_text = enc.decode(generated_tokens) # decode tokens to text
#     print(f"=== GENERATED SEQUENCE {i+1} ===")
#     print(generated_text)
#     print()