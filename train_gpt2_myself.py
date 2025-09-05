import os
import math
import time
import inspect # for introspection
from dataclasses import dataclass # for configuration
import torch # pytorch
import torch.nn as nn # neural network modules
from torch.nn import functional as F # for activation functions etc.
from hellaswag import render_example, iterate_examples # for HellaSwag eval
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
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        if set.current_position + B * T + 1 >= len(self.tokens):
            self.current_position = 0
        return x, y
    
# ------------------------------------------------------------------------------

train_loader = DataLoaderLite(B=4, T=32) # create a data loader

# generation hyperparameters
num_return_sequences = 3
max_length = 100

# get a data batch
enc = tiktoken.get_encoding("gpt2") # get GPT-2 tokenizer
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000] # use only the first 10,000 characters
print("example text: ", text[:1000]) # print the first 200 characters
tokens = enc.encode(text) # encode text to tokens
print(f"length of dataset in characters: {len(text)}")
print(f"length of dataset in tokens: {len(tokens)}")
B, T = 4, 32 # batch size, sequence length
buf = torch.tensor(tokens[:B*T + 1]) # create a buffer of tokens
x = buf[:-1].view(B, T) # input tokens
y = buf[1:].view(B, T) # target tokens
print("x: ", x.shape, x.dtype) # (B, T) in torch.long
print("y: ", y.shape, y.dtype) # (B, T) in torch

# to GPU
x = x.to('cuda')
y = y.to('cuda')

# create the model
# model = GPT.from_pretrained('gpt2') # load pretrained model
model = GPT(GPTConfig()) # load a fresh model
print("number of parameters: %.2fM" % (sum(p.numel() for p in model.parameters())/1e6,))
model.to('cuda') # move model to GPU

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for i in range(50): # 10 iterations of training
    optimizer.zero_grad(set_to_none=True) # reset gradients
    logits, loss = model(x, y) # forward pass
    loss.backward() # backward pass
    optimizer.step() # update parameters
    print(f"iteration {i+1}, loss: {loss.item():.4f}") # print loss

# toy model
# tokens = enc.encode("Explain the moon landing to a 6 year old in a few sentences.") # encode input text
# tokens = torch.tensor(tokens, dtype=torch.long) # (7,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 7)  (batch size, sequence length)
# x = tokens.to('cuda') # move tokens to GPU

# gernerate text
torch.manual_seed(42) # for reproducibility
torch.cuda.manual_seed(42) # for reproducibility
with torch.no_grad(): # no grad context
    for _ in range(max_length):
        logits, _ = model(x.to('cuda')) # (5, T, 50257)
        logits = logits[:, -1, :] # (5, 50257) take the last token
        probs = F.softmax(logits, dim=-1) # (5, 50257) softmax to get probabilities
        topk_probs, topk_indices = torch.topk(probs, k=10, dim=-1) # (5, 10) get top 10 probabilities and indices
        ix = torch.multinomial(topk_probs, num_samples=1) # (5, 1) sample from the top 10
        next_token = torch.gather(topk_indices, -1, ix) # (5, 1) get the actual token
        x = torch.cat((x, next_token), dim=1) # (5, T+1) append to the sequence

# # print generated text
# for i in range(num_return_sequences):
#     tokens = x[i, :max_length + 7].tolist() # get tokens
#     text = enc.decode(tokens) # decode tokens to text
#     print(">", text) # print generated text

# print generated text
for i in range(num_return_sequences):
    generated_tokens = x[i].cpu().numpy() # move to cpu and convert to numpy
    generated_text = enc.decode(generated_tokens) # decode tokens to text
    print(f"=== GENERATED SEQUENCE {i+1} ===")
    print(generated_text)
    print()