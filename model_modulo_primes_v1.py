import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_FILENAME = "model_modulo_primes_v1.pth"
FORCE_TRAINING = True

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

# Vocabulaire enrichi avec ',', '+', '-'
vocab_tokens = sorted(list(token_values.keys()))
vocab_tokens.extend([",", "+", "-"])
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

comma_idx = token_to_idx[","]
plus_idx = token_to_idx["+"]
minus_idx = token_to_idx["-"]

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

# --- CONSTRUCTION DU CORPUS AVEC MODULO 6 (+ et -) ---
print("Construction du corpus avec balisage Modulo 6 (+ pour 1 mod 6, - pour 5 mod 6) ...")
real_tokens = [t for t in vocab_tokens if t not in [",", "+", "-"]]

primes = [p for p in range(2, 50000) if is_prime(p)]
corpus_tokens = []

for p in primes:
    p_bin = bin(p)[2:]
    path = find_first_valid_path(p_bin, real_tokens)
    
    if path is not None:
        # Ajout du signe modulo 6 pour les nombres > 3
        if p > 3:
            if p % 6 == 1:
                corpus_tokens.append(plus_idx)
            elif p % 6 == 5:
                corpus_tokens.append(minus_idx)
                
        # Ajout des briques du nombre
        for idx_t, t in enumerate(path):
            corpus_tokens.append(token_to_idx[t])
            if idx_t < len(path) - 1:
                corpus_tokens.append(comma_idx)
                
        corpus_tokens.append(comma_idx)

data = torch.tensor(corpus_tokens, dtype=torch.long)
print(f"Taille du corpus continu avec modulos : {len(data)} tokens.")

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
    def generate_stream(self, max_tokens=400, temperature=0.7, top_k=5):
        idx = torch.tensor([[token_to_idx[real_tokens[0]]]], dtype=torch.long)
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

# --- ENTRAÎNEMENT ---
model = TokenTransformer(vocab_size, block_size=16)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

if os.path.exists(MODEL_FILENAME) and not FORCE_TRAINING:
    print(f"Chargement du modèle depuis '{MODEL_FILENAME}'...")
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()
else:
    print("Entraînement du modèle 'Modulo Primes' (600 époques)...")
    model.train()
    for epoch in range(600):
        ix = torch.randint(len(data) - 16, (64,))
        x = torch.stack([data[i:i+16] for i in ix])
        y = torch.stack([data[i+1:i+17] for i in ix])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Modèle sauvegardé dans '{MODEL_FILENAME}'.")

# --- ARBITRE PYTHON : VERIFICATION DES MODULOS ---
print("\n=== ÉVALUATION DU MODÈLE MODULO PRIMES ===")
raw_tokens = model.generate_stream(max_tokens=400)

generated_text = "".join([idx_to_token[i] for i in raw_tokens])
print(f"Extrait du flux brut généré :\n{generated_text[:250]}...\n")

# Découpage par nombres (séparés par des virgules)
chunks = generated_text.split(",")
valid_count = 0
total_checked = 0

print("Analyse des nombres générés et de leur cohérence modulo 6 :\n")
for chunk in chunks:
    if not chunk: continue
    
    # Vérification de la présence d'un signe + ou - au début
    sign = None
    num_str = chunk
    if chunk.startswith("+"):
        sign = "+"
        num_str = chunk[1:]
    elif chunk.startswith("-"):
        sign = "-"
        num_str = chunk[1:]
        
    clean_bin = num_str.replace("+", "").replace("-", "")
    if not clean_bin: continue
    
    try:
        val = int(clean_bin, 2)
        is_p = is_prime(val)
        
        # Calcul du modulo 6 réel
        actual_mod = val % 6
        expected_sign = "+" if actual_mod == 1 else ("-" if actual_mod == 5 else "autre")
        
        match_sign = (sign == expected_sign) if sign else (val <= 3)
        
        if is_p:
            total_checked += 1
            if match_sign:
                valid_count += 1
                print(f"🌟 {sign}{clean_bin} ({val}) -> Premier : OUI | Modulo 6 correct ({actual_mod})")
            else:
                print(f"❌ {sign}{clean_bin} ({val}) -> Premier : OUI | ERREUR Modulo (attendu {expected_sign}, trouvé {sign})")
        else:
            print(f"⚠️ {sign}{clean_bin} ({val}) -> N'est pas premier")
            
    except ValueError:
        pass

if total_checked > 0:
    print(f"\n🎯 Taux de cohérence du Modulo 6 sur les premiers générés : {valid_count}/{total_checked} ({(valid_count/total_checked)*100:.1f}%)")
else:
    print("\nAucun nombre premier valide n'a pu être extrait pour le test.")