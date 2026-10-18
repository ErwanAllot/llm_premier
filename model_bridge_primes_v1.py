import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_FILENAME = "model_bridge_primes_v1.pth"
FORCE_TRAINING = True  # True pour entraîner de zéro cette nouvelle grammaire

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

# Vocabulaire enrichi avec '(', ')', ','
vocab_tokens = sorted(list(token_values.keys()))
vocab_tokens.extend([",", "(", ")"])
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

comma_idx = token_to_idx[","]
lparen_idx = token_to_idx["("]
rparen_idx = token_to_idx[")"]

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

# --- CONSTRUCTION DU CORPUS DE PONTS PARENTHÉSÉS ---
print("Construction du corpus des nombres premiers avec ponts (...) ...")
real_tokens = [t for t in vocab_tokens if t not in [",", "(", ")"]]

# 1. Génération et décomposition des nombres premiers
primes = [p for p in range(2, 50000) if is_prime(p)]
decomposed_primes = []

for p in primes:
    path = find_first_valid_path(bin(p)[2:], real_tokens)
    if path is not None:
        decomposed_primes.append(path)

# 2. Assemblage du flux avec détection des ponts entre P1 et P2
corpus_tokens = []
i = 0
while i < len(decomposed_primes) - 1:
    p1_path = decomposed_primes[i]
    p2_path = decomposed_primes[i+1]
    
    bridge_found = False
    # On vérifie si P1 et P2 ont au moins 2 tokens pour faire b_a1(...)b_b2
    if len(p1_path) >= 2 and len(p2_path) >= 2:
        b_b1 = p1_path[-1]  # Dernier token de P1
        b_a2 = p2_path[0]   # Premier token de P2
        
        # Test si la concaténation binaire des deux tokens forme un premier
        concat_bin = b_b1 + b_a2
        try:
            concat_val = int(concat_bin, 2)
            if is_prime(concat_val):
                bridge_found = True
        except ValueError:
            pass

    if bridge_found:
        # Syntaxe : b_a1(b_b1, b_a2)b_b2,
        b_a1_tokens = p1_path[:-1]
        b_b2_tokens = p2_path[1:]
        
        # Partie initiale de P1
        for t in b_a1_tokens:
            corpus_tokens.append(token_to_idx[t])
            
        # Ouverture du pont (
        corpus_tokens.append(lparen_idx)
        
        # b_b1 , b_a2
        corpus_tokens.append(token_to_idx[b_b1])
        corpus_tokens.append(comma_idx)
        corpus_tokens.append(token_to_idx[b_a2])
        
        # Fermeture du pont )
        corpus_tokens.append(rparen_idx)
        
        # Reste de P2
        for t in b_b2_tokens:
            corpus_tokens.append(token_to_idx[t])
            
        corpus_tokens.append(comma_idx)
        i += 2  # On a consommé P1 et P2 conjointement
    else:
        # Flux normal pour P1 : t1 t2 ... ,
        for t in p1_path:
            corpus_tokens.append(token_to_idx[t])
        corpus_tokens.append(comma_idx)
        i += 1

data = torch.tensor(corpus_tokens, dtype=torch.long)
print(f"Taille du corpus continu avec ponts : {len(data)} tokens.")

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
        # Démarrage avec le premier token brique du vocabulaire
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

# --- ENTRAÎNEMENT DE ZÉRO ---
model = TokenTransformer(vocab_size, block_size=16)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

if os.path.exists(MODEL_FILENAME) and not FORCE_TRAINING:
    print(f"Chargement du modèle depuis '{MODEL_FILENAME}'...")
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()
else:
    print("Entraînement du modèle 'Bridge Primes' en cours (600 époques)...")
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

# --- ARBITRE PYTHON : VERIFICATION DE LA SYNTAXE ET DES PONTS ---
print("\n=== ÉVALUATION DES PONTS GÉNÉRÉS EN PARENTHÈSES ===")
raw_tokens = model.generate_stream(max_tokens=400)

generated_text = "".join([idx_to_token[i] for i in raw_tokens])
print(f"Extrait du flux brut généré :\n{generated_text[:200]}...\n")

# Extraction et validation des ponts (...)
import re
bridges = re.findall(r'\(([^,]+),([^)]+)\)', generated_text)

valid_bridges = 0
total_bridges = len(bridges)

print(f"Nombre de structures de pont (t1, t2) identifiées : {total_bridges}\n")

for t1, t2 in bridges:
    concat_bin = t1 + t2
    try:
        val = int(concat_bin, 2)
        if is_prime(val):
            valid_bridges += 1
            print(f"🌟 PONT VALIDE : ({t1}, {t2}) -> Concat binaire = {concat_bin} = {val} | Premier : OUI")
        else:
            print(f"❌ Pont invalide : ({t1}, {t2}) -> Concat binaire = {concat_bin} = {val} | Premier : NON")
    except ValueError:
        print(f"⚠️ Erreur de décodage pour le pont : ({t1}, {t2})")

if total_bridges > 0:
    print(f"\n🎯 Taux de validité arithmétique des ponts : {valid_bridges}/{total_bridges} ({(valid_bridges/total_bridges)*100:.1f}%)")
else:
    print("\nAucun pont parenthésé complet n'a été détecté dans cet extrait.")