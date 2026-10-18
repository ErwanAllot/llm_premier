import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_FILENAME = "model_comma_stream_v1.pth"
FORCE_TRAINING = False  # Mettez à True pour lancer un nouvel entraînement

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

token_values = {}
for val, norm, barre in BASE_TOKENS:
    token_values[norm] = ("prime", val)
    token_values[barre] = ("barre", val)

vocab_tokens = sorted(list(token_values.keys()))
vocab_tokens.append(",")
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}
comma_idx = token_to_idx[","]

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

# --- CORPUS EN FLUX CONTINU ---
print("Construction du corpus en flux continu...")
corpus_tokens = []
real_tokens = [t for t in vocab_tokens if t != ","]

for p in range(2, 50000):
    if is_prime(p):
        s = bin(p)[2:]
        path = find_first_valid_path(s, real_tokens)
        if path is not None:
            corpus_tokens.extend([token_to_idx[t] for t in path] + [comma_idx])

data = torch.tensor(corpus_tokens, dtype=torch.long)
print(f"Taille du corpus continu : {len(data)} tokens.")

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
    def generate_stream(self, max_tokens=300, temperature=0.8, top_k=5):
        idx = torch.randint(0, vocab_size - 1, (1, 1))
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

# --- CHARGEMENT OU ENTRAÎNEMENT ---
model = TokenTransformer(vocab_size, block_size=16)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

if os.path.exists(MODEL_FILENAME) and not FORCE_TRAINING:
    print(f"Chargement du modèle depuis '{MODEL_FILENAME}'...")
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()
else:
    print("Entraînement du modèle...")
    model.train()
    for epoch in range(400):
        ix = torch.randint(len(data) - 16, (64,))
        x = torch.stack([data[i:i+16] for i in ix])
        y = torch.stack([data[i+1:i+17] for i in ix])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Modèle sauvegardé dans '{MODEL_FILENAME}'.")

# --- GÉNÉRATION ET ÉVALUATION STRATIFIÉE PAR TAILLE ---
print("\n=== ÉVALUATION PAR NIVEAU DE LONGUEUR DE TOKENS ===")
stats_by_length = {1: [], 2: [], 3: [], 4: [], 5: []}
target_per_level = 5

# On boucle jusqu'à avoir récolté assez d'exemples pour chaque niveau de 1 à 5 tokens
while any(len(v) < target_per_level for v in stats_by_length.values()):
    raw_tokens_idx = model.generate_stream(max_tokens=400)
    
    # Découpage du flux par les virgules
    current_number = []
    numbers_tokens = []
    for idx in raw_tokens_idx:
        if idx == comma_idx:
            if current_number:
                numbers_tokens.append(current_number)
                current_number = []
        else:
            current_number.append(idx)
            
    # Tri par longueur
    for t_list in numbers_tokens:
        length = len(t_list)
        if 1 <= length <= 5 and len(stats_by_length[length]) < target_per_level:
            token_strs = [idx_to_token[i] for i in t_list]
            bin_str = "".join(token_strs)
            try:
                dec_val = int(bin_str, 2)
                is_p = is_prime(dec_val)
                stats_by_length[length].append((token_strs, dec_val, is_p))
            except ValueError:
                continue

# --- AFFICHAGE DES RÉSULTATS ---
total_success = 0
total_generated = 0

for length in range(1, 6):
    print(f"\n--- Niveau : {length} token(s) ---")
    success_in_level = 0
    for token_strs, dec_val, is_p in stats_by_length[length]:
        status = "🌟 OUI" if is_p else "non"
        if is_p: success_in_level += 1
        print(f"  Tokens: {token_strs} | Décimal: {dec_val} | Premier: {status}")
    
    rate = (success_in_level / target_per_level) * 100
    print(f"👉 Succès pour {length} token(s) : {success_in_level}/{target_per_level} ({rate:.1f}%)")
    total_success += success_in_level
    total_generated += target_per_level

global_rate = (total_success / total_generated) * 100
print(f"\n==========================================")
print(f"🎯 TAUX DE SUCCÈS GLOBAL : {total_success}/{total_generated} ({global_rate:.1f}%)")
print(f"==========================================")