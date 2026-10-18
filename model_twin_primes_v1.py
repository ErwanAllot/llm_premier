import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_FILENAME = "model_twin_primes_v1.pth"
FORCE_TRAINING = True  # True pour lancer l'entraînement musclé

def is_prime(num):
    if num < 2: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

# Briques élémentaires
BASE_TOKENS = [
    (2, "10", "01"), (3, "11", "00"), (5, "101", "010"),
    (7, "111", "000"), (17, "10001", "01110"),
    (19, "10011", "01100"), (409, "110011001", "001100110")
]

token_values = {}
for val, norm, barre in BASE_TOKENS:
    token_values[norm] = ("prime", val)
    token_values[barre] = ("barre", val)

vocab_tokens = sorted(list(token_values.keys()))
vocab_tokens.extend([",", "/"])  # Séparateur et Amorce
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

comma_idx = token_to_idx[","]
start_idx = token_to_idx["/"]

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

# --- CONSTRUCTION D'UN CORPUS ÉLARGI DE JUMEAUX (JUSQU'À 150 000) ---
print("Construction du corpus élargi de jumeaux avec '/' (jusqu'à 150 000)...")
corpus_tokens = [start_idx]
real_tokens = [t for t in vocab_tokens if t not in [",", "/"]]

p = 3
while p < 150000:
    if is_prime(p):
        if p == 3 and is_prime(5) and is_prime(7):
            for val in [3, 5, 7]:
                path = find_first_valid_path(bin(val)[2:], real_tokens)
                if path:
                    corpus_tokens.extend([token_to_idx[t] for t in path] + [comma_idx])
            p = 7
        elif is_prime(p + 2):
            for val in [p, p + 2]:
                path = find_first_valid_path(bin(val)[2:], real_tokens)
                if path:
                    corpus_tokens.extend([token_to_idx[t] for t in path] + [comma_idx])
    p += 2

data = torch.tensor(corpus_tokens, dtype=torch.long)
print(f"Taille du nouveau corpus continu : {len(data)} tokens.")

# --- ARCHITECTURE DU TRANSFORMER ---
class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=4, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(block_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        device = idx.device
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.token_embedding(idx) + self.pos_embedding(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        return self.fc_out(self.transformer(x, mask=mask, is_causal=True))

    @torch.no_grad()
    def generate_stream(self, max_tokens=300, temperature=0.5, top_k=5): # Température baissée à 0.5
        idx = torch.tensor([[start_idx]], dtype=torch.long)
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx.squeeze().tolist()

# --- ENTRAÎNEMENT MUSCLÉ (800 ÉPOQUES) ---
model = TokenTransformer(vocab_size, block_size=16)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

if os.path.exists(MODEL_FILENAME) and not FORCE_TRAINING:
    print(f"Chargement du modèle pré-entraîné depuis '{MODEL_FILENAME}'...")
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()
else:
    print("Entraînement renforcé du modèle 'Jumeaux' (800 époques)...")
    model.train()
    for epoch in range(800):
        ix = torch.randint(len(data) - 16, (64,))
        x = torch.stack([data[i:i+16] for i in ix])
        y = torch.stack([data[i+1:i+17] for i in ix])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Modèle sauvegardé dans '{MODEL_FILENAME}'.")

# --- ARBITRE PYTHON : ANALYSE DES PAIRES ---
print("\n=== ÉVALUATION DE LA GÉNÉRATION DE JUMEAUX (CORPUS ÉLARGI) ===")
raw_tokens = model.generate_stream(max_tokens=400)

numbers_list = []
current = []
for tok_i in raw_tokens:
    if tok_i == start_idx:
        continue
    elif tok_i == comma_idx:
        if current:
            token_strs = [idx_to_token[i] for i in current]
            bin_str = "".join(token_strs)
            try:
                dec_val = int(bin_str, 2)
                numbers_list.append((token_strs, dec_val))
            except ValueError:
                pass
            current = []
    else:
        current.append(tok_i)

print(f"Nombres générés dans la séquence : {[n[1] for n in numbers_list[:15]]}...\n")

valid_twin_pairs = 0
total_pairs_checked = 0

for i in range(len(numbers_list) - 1):
    n1_toks, n1_val = numbers_list[i]
    n2_toks, n2_val = numbers_list[i+1]
    
    p1_ok = is_prime(n1_val)
    p2_ok = is_prime(n2_val)
    diff = abs(n2_val - n1_val)
    
    total_pairs_checked += 1
    if p1_ok and p2_ok and diff == 2:
        valid_twin_pairs += 1
        print(f"🌟 PAIRE DE JUMEAUX VALIDE : {n1_val} ({n1_toks}) <-> {n2_val} ({n2_toks}) | Écart = {diff}")
    elif p1_ok and p2_ok:
        print(f"⚠️  Premiers non-jumeaux : {n1_val} <-> {n2_val} | Écart = {diff}")
    else:
        print(f"❌  Erreur (Non premiers) : {n1_val} <-> {n2_val}")

if total_pairs_checked > 0:
    print(f"\n🎯 Taux de succès sur les paires consécutives : {valid_twin_pairs}/{total_pairs_checked} ({(valid_twin_pairs/total_pairs_checked)*100:.1f}%)")