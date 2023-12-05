import logging
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences to process in parallel
block_size = 8  # maximum context length
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "mps"
eval_iters = 200
n_embed = 32


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention score - "affinities"
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        vec = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(vec)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by ReLu."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads.
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x1 = x + self.sa(x)
        x2 = x1 + self.ffwd(x1)
        return x2


class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
            Block(n_embed, n_head=4),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x1 = tok_emb + pos_emb
        x2 = self.blocks(x1)
        logits = self.lm_head(x2)  # (B,T,vocab_size)
        if targets is None:
            logits2 = logits
            loss = None
        else:
            B, T, C = logits.shape
            logits2 = logits.view(B*T, C)
            targets2 = targets.view(B*T)
            loss = F.cross_entropy(logits2, targets2)
        return logits2, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predicted logits
            logits, loss = self(idx_cond)
            # foxus only on the last time step
            logits_last = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits_last, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx


logging.basicConfig(level="INFO")
torch.manual_seed(1337)
logging.info("started")
with open("input.txt", "r", encoding="utf-8") as fp:
    text = fp.read()
logging.info(len(text))
chars = sorted(list(set(text)))
vocab_size = len(chars)
logging.info(f"vocabulary is {vocab_size}: {''.join(chars)}")
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda x: ''.join([itos[i] for  i in x])
s = "hii there"
logging.info(f"encode({s}) = {encode(s)}")
logging.info(f"decode({encode(s)}) = {decode(encode(s))}")
data = torch.tensor(encode(text), dtype=torch.long)
logging.info(f"data shape: {data.shape}")
logging.info(f"data type: {data.dtype}")
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
logging.info(f"train: {train_data.shape}, validation: {val_data.shape}")
xb, yb = get_batch("train")
logging.info(f"inputs: {xb.shape} targets: {yb.shape}")
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        logging.info(f"{b=} {t=} when input is {context.tolist()} the target is {target}")
model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)
logging.info(f"output: {logits.shape}")
logging.info(f"loss: {loss}")
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
tokids = m.generate(idx, max_new_tokens=100)
logging.debug(f"generated token ids: {tokids.tolist()}")
logging.info(decode(tokids[0].tolist()))
# train the model
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        logging.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
logging.info(f"loss after training: {loss.item()}")
tokids = m.generate(idx, max_new_tokens=400)
logging.info(f"output after training: {decode(tokids[0].tolist())}")
logging.info("completed")