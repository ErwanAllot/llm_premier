import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. GÉNÉRATION DU CORPUS DE SYNTAXE PURE ---

BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

VALID_BRICKS = set()
for val, b1, b2 in BASE_TOKENS:
    VALID_BRICKS.add(b1)
    VALID_BRICKS.add(b2)

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def decompose_number(n_bin):
    if n_bin in VALID_BRICKS:
        return [n_bin]
    max_len = max(len(b) for b in VALID_BRICKS)
    for i in range(min(len(n_bin), max_len), 0, -1):
        prefix = n_bin[:i]
        if prefix in VALID_BRICKS:
            suffix = n_bin[i:]
            if not suffix:
                return [prefix]
            sub_res = decompose_number(suffix)
            if sub_res is not None:
                return [prefix] + sub_res
    return None

def get_tag_by_length(n):
    mapping = {2: ('A', 'a'), 3: ('B', 'b'), 4: ('C', 'c'), 5: ('D', 'd'), 6: ('E', 'e')}
    return mapping.get(n, (None, None))

def get_modulation(p):
    r6 = p % 6
    mod6_str = "+" if r6 == 1 else ("-" if r6 == 5 else "&")
    r4 = p % 4
    mod4_str = "*" if r4 == 1 else ("%" if r4 == 3 else "&")
    return f"{mod6_str}, {mod4_str}"

def generate_pure_syntax_corpus(max_primes=800):
    print(f"Génération du corpus de syntaxe pure pour les {max_primes} premiers nombres...")
    primes = []
    c = 2
    while len(primes) < max_primes:
        if is_prime(c):
            primes.append(c)
        c += 1

    prime_data = []
    global_tokens = []

    for p in primes:
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        prime_data.append({'p': p, 'mod_str': get_modulation(p), 'bricks': bricks})
        if bricks:
            for b in bricks:
                global_tokens.append({'token': b})

    open_tags = [[] for _ in range(len(global_tokens))]
    close_tags = [[] for _ in range(len(global_tokens))]

    for N in range(2, 7):
        for i in range(len(global_tokens) - N + 1):
            window = global_tokens[i:i+N]
            concat_bin = "".join([t['token'] for t in window])
            try:
                val = int(concat_bin, 2)
                if is_prime(val):
                    open_char, close_char = get_tag_by_length(N)
                    if open_char and open_char not in open_tags[i]:
                        open_tags[i].append(open_char)
                    if close_char and close_char not in close_tags[i + N - 1]:
                        close_tags[i + N - 1].append(close_char)
            except ValueError:
                pass

    syntax_blocks = []
    token_global_idx = 0
    for item in prime_data:
        mod_str = item['mod_str']
        bricks = item['bricks']
        if bricks is None:
            bricks_str = "(néant)"
        else:
            brick_strs = []
            for b in bricks:
                o_str = "".join(open_tags[token_global_idx])
                c_str = "".join(close_tags[token_global_idx])
                token_global_idx += 1
                brick_strs.append(f"{o_str}({b}){c_str}")
            bricks_str = "".join(brick_strs)
        
        # Format strict demandé : <[modulation]briques> collés les uns aux autres
        syntax_blocks.append(f"<[{mod_str}]{bricks_str}>")
    
    return "".join(syntax_blocks)

# --- 2. ARCHITECTURE MINI-GPT ---

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GigogneLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 3. ENTRAÎNEMENT ET SAUVEGARDE ---

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Utilisation du device : {device}")

    corpus_text = generate_pyure_syntax_corpus = generate_pure_syntax_corpus(max_primes=800)
    
    chars = sorted(list(set(corpus_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(corpus_text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]

    block_size = 64
    batch_size = 32
    n_embd = 128
    n_head = 4
    n_layer = 4
    max_iters = 1000
    learning_rate = 3e-4

    def get_batch():
        ix = torch.randint(len(train_data) - block_size, (batch_size,))
        x = torch.stack([train_data[i:i+block_size] for i in ix])
        y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    model = GigogneLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Début de l'entraînement du modèle sur le ruban pur...")
    model.train()
    for iter in range(max_iters):
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 200 == 0 or iter == max_iters - 1:
            print(f"Itération {iter}/{max_iters} - Perte (Loss): {loss.item():.4f}")

    model_path = "prime_gigogne_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé dans '{model_path}' !")

    # Test de génération avec une amorce pure
    model.eval()
    context = torch.tensor(encode("<[&, &]AC(10)><"), dtype=torch.long, device=device).unsqueeze(0)
    generated = decode(model.generate(context, max_new_tokens=120)[0].tolist())
    
    print("\n--- TEST DE GÉNÉRATION EN SORTIE DU MODÈLE ---")
    print(generated)

if __name__ == "__main__":
    main()