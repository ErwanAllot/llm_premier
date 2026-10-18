import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- CONFIGURATION ---
MODEL_FILENAME = "model_macro_bridge_primes_v1.pth"
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

# Vocabulaire enrichi avec '(', ')', '[', ']', ','
vocab_tokens = sorted(list(token_values.keys()))
vocab_tokens.extend([",", "(", ")", "[", "]"])
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}

comma_idx = token_to_idx[","]
lparen_idx = token_to_idx["("]
rparen_idx = token_to_idx[")"]
lbrack_idx = token_to_idx["["]
rbrack_idx = token_to_idx["]"]

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

# --- CONSTRUCTION DU CORPUS MACRO-PONTS [...] ---
print("Construction du corpus avec ponts (...) et macro-extensions [...] ...")
real_tokens = [t for t in vocab_tokens if t not in [",", "(", ")", "[", "]"]]

primes = [p for p in range(2, 50000) if is_prime(p)]
decomposed_primes = []

for p in primes:
    path = find_first_valid_path(bin(p)[2:], real_tokens)
    if path is not None:
        decomposed_primes.append(path)

corpus_tokens = []
i = 0
while i < len(decomposed_primes) - 1:
    p1_path = decomposed_primes[i]
    p2_path = decomposed_primes[i+1]
    
    bridge_found = False
    if len(p1_path) >= 2 and len(p2_path) >= 2:
        b_b1 = p1_path[-1]  # Dernier token de P1
        b_a2 = p2_path[0]   # Premier token de P2
        
        concat_bin = b_b1 + b_a2
        try:
            if is_prime(int(concat_bin, 2)):
                bridge_found = True
        except ValueError:
            pass

    if bridge_found:
        b_a1_tokens = p1_path[:-1]
        b_b2_tokens = p2_path[1:]
        
        # Recherche d'une extension vers la droite à partir du token d'ouverture (b_b1)
        # On teste l'ajout successif des tokens suivants de P2
        extension_len = 0
        current_bin = b_b1 + b_a2
        
        for ext_idx, ext_tok in enumerate(b_b2_tokens):
            current_bin += ext_tok
            try:
                if is_prime(int(current_bin, 2)):
                    extension_len = ext_idx + 1  # Nombre de tokens supplémentaires absorbés
            except ValueError:
                pass

        # 1. Partie initiale de P1
        for t in b_a1_tokens:
            corpus_tokens.append(token_to_idx[t])
            
        # 2. Si une extension premier est trouvée, on ouvre le crochet '['
        if extension_len > 0:
            corpus_tokens.append(lbrack_idx)
            
        # 3. Structure du pont parenthésé '(' b_b1 ',' b_a2 ')'
        corpus_tokens.append(lparen_idx)
        corpus_tokens.append(token_to_idx[b_b1])
        corpus_tokens.append(comma_idx)
        corpus_tokens.append(token_to_idx[b_a2])
        corpus_tokens.append(rparen_idx)
        
        # 4. Ajout des tokens étendus à droite
        for k in range(extension_len):
            corpus_tokens.append(token_to_idx[b_b2_tokens[k]])
            
        # 5. Fermeture du crochet ']' si extension présente
        if extension_len > 0:
            corpus_tokens.append(rbrack_idx)
            
        # 6. Tokens restants de P2 non absorbés par le crochet
        for k in range(extension_len, len(b_b2_tokens)):
            corpus_tokens.append(token_to_idx[b_b2_tokens[k]])
            
        corpus_tokens.append(comma_idx)
        i += 2
    else:
        for t in p1_path:
            corpus_tokens.append(token_to_idx[t])
        corpus_tokens.append(comma_idx)
        i += 1

data = torch.tensor(corpus_tokens, dtype=torch.long)
print(f"Taille du corpus continu avec crochets : {len(data)} tokens.")

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
    def generate_stream(self, max_tokens=500, temperature=0.7, top_k=5):
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
    print("Entraînement du modèle 'Macro Bridge Primes' (700 époques)...")
    model.train()
    for epoch in range(700):
        ix = torch.randint(len(data) - 16, (64,))
        x = torch.stack([data[i:i+16] for i in ix])
        y = torch.stack([data[i+1:i+17] for i in ix])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_FILENAME)
    print(f"Modèle sauvegardé dans '{MODEL_FILENAME}'.")

# --- ARBITRE PYTHON : VERIFICATION DES PONTS () ET EXTENSIONS [...] ---
print("\n=== ÉVALUATION DES MACRO-STRUCTURES GÉNÉRÉES ===")
raw_tokens = model.generate_stream(max_tokens=500)

generated_text = "".join([idx_to_token[i] for i in raw_tokens])
print(f"Extrait du flux brut généré :\n{generated_text[:250]}...\n")

# 1. Évaluation des ponts ()
bridges = re.findall(r'\(([^,]+),([^)]+)\)', generated_text)
valid_bridges = 0
for t1, t2 in bridges:
    try:
        if is_prime(int(t1 + t2, 2)):
            valid_bridges += 1
    except ValueError:
        pass

if len(bridges) > 0:
    print(f"🎯 Ponts () valides : {valid_bridges}/{len(bridges)} ({(valid_bridges/len(bridges))*100:.1f}%)")

# 2. Évaluation des macro-extensions [...]
macro_pattern = re.findall(r'\[\(([^,]+),([^)]+)\)([^\]]*)\]', generated_text)
valid_macros = 0
total_macros = len(macro_pattern)

print(f"\nNombre de macro-extensions [...] identifiées : {total_macros}\n")

for t1, t2, extension in macro_pattern:
    full_bin = t1 + t2 + extension
    try:
        val = int(full_bin, 2)
        if is_prime(val):
            valid_macros += 1
            print(f"🌟 MACRO-EXTENSION VALIDE : [({t1},{t2}){extension}] -> Concat binaire = {full_bin} = {val} | Premier : OUI")
        else:
            print(f"❌ Macro-extension invalide : [({t1},{t2}){extension}] -> Concat binaire = {full_bin} = {val} | Premier : NON")
    except ValueError:
        print(f"⚠️ Erreur de décodage pour la macro : [({t1},{t2}){extension}]")

if total_macros > 0:
    print(f"\n🎯 Taux de validité des macro-extensions [...] : {valid_macros}/{total_macros} ({(valid_macros/total_macros)*100:.1f}%)")
else:
    print("\nAucune macro-extension [...] complète détectée dans cet extrait.")