import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

def is_prime(num):
    if num < 2: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

BASE_TOKENS = [
    (2, "10", "01"), (3, "11", "00"), (5, "101", "010"),
    (7, "111", "000"), (17, "10001", "01110"),
    (19, "10011", "01100"), (409, "110011001", "001100110")
]

# Notre alphabet de briques + un token spécial de fin '<EOS>'
vocab_tokens = sorted(list(set([t for _, t, _ in BASE_TOKENS] + [b for _, _, b in BASE_TOKENS])))
vocab_tokens.append("<EOS>")
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}
eos_idx = token_to_idx["<EOS>"]

def find_first_valid_path(sub_s, blks):
    sorted_blks = sorted(list(set(blks)), key=len, reverse=True)
    memo = {}
    def backtrack(s):
        if not s: return []
        if s in memo: return memo[s]
        for b in sorted_blks:
            if b and s.startswith(b):
                res = backtrack(s[len(b):])
                if res is not None:
                    path = [b] + res
                    memo[s] = path
                    return path
        memo[s] = None
        return None
    return backtrack(sub_s)

# --- CORPUS STRUCTURÉ AVEC DÉLIMITATEUR ---
print("Construction du corpus structuré avec <EOS>...")
corpus_sequences = []
# for p in range(2, 500000):
for p in range(2, 100000):
    if is_prime(p):
        s = bin(p)[2:]
        # On teste uniquement avec les vraies briques (sans le mot <EOS>)
        real_tokens = [t for t in vocab_tokens if t != "<EOS>"]
        path = find_first_valid_path(s, real_tokens)
        if path is not None:
            # On ajoute chaque nombre comme une phrase complète suivie de <EOS>
            corpus_sequences.append([token_to_idx[t] for t in path] + [eos_idx])

data = []
for seq in corpus_sequences:
    data.extend(seq)
data = torch.tensor(data, dtype=torch.long)
print(f"Taille du corpus structuré : {len(data)} tokens.")

# --- ARCHITECTURE DU TRANSFORMER ---
class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=4, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        return self.fc_out(self.transformer(x, mask=mask, is_causal=True))

    @torch.no_grad()
    def generate(self, max_new_tokens=8):
        # On commence la génération à partir d'un token aléatoire de l'alphabet (hors EOS)
        start_t = torch.randint(0, vocab_size - 1, (1, 1))
        idx = start_t
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            # Si le modèle génère le token EOS, on arrête la phrase !
            if next_idx.item() == eos_idx:
                break
                
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

block_size = 8
batch_size = 64
model = TokenTransformer(vocab_size, block_size=block_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("Entraînement du modèle structuré...")
model.train()
for epoch in range(600):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    optimizer.zero_grad()
    loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()

# --- TEST ---
model.eval()
print("\n=== GÉNÉRATION AVEC DÉLIMITATEUR <EOS> ===")
num_samples = 40
valid_prime_count = 0

for _ in range(num_samples):
    generated_indices = model.generate(max_new_tokens=6)
    tokens_sequence = [idx_to_token[idx.item()] for idx in generated_indices[0]]
    
    candidate_bin = "".join(tokens_sequence)
    decimal_val = int(candidate_bin, 2)
    is_p = is_prime(decimal_val)
    if is_p: valid_prime_count += 1
        
    print(f"Tokens : {tokens_sequence} -> Binaire: {candidate_bin} | Décimal: {decimal_val} | Premier: {'🌟 OUI' if is_p else 'non'}")

print(f"\nTaux de nombres premiers générés (avec structures propres) : {(valid_prime_count / num_samples) * 100:.1f}%")