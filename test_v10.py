import os
import torch
import torch.nn as nn
from sympy import isprime

# =====================================================================
# 1. EXTRACTION DE VOS TOKENS IRRÉDUCTIBLES (Votre Code Exact)
# =====================================================================

NB_PREMIERS_CIBLE = 10000

def check_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def get_barre(s):
    return "".join('1' if c == '0' else '0' for c in s)

print(f"Génération des {NB_PREMIERS_CIBLE} premiers nombres premiers...")
primes = []
n = 2
while len(primes) < NB_PREMIERS_CIBLE:
    if check_prime(n):
        primes.append(n)
    n += 1

tokens_bin = []
token_set = set()

def can_partition(sub_s, blks):
    if not sub_s: return True
    unique_blks = sorted(list(set(blks)), key=len, reverse=True)
    for b in unique_blks:
        if b and sub_s.startswith(b):
            if can_partition(sub_s[len(b):], blks):
                return True
    return False

for p in primes:
    s = bin(p)[2:]
    blocks = []
    for t in tokens_bin:
        blocks.append(t)
        blocks.append(get_barre(t))
    
    if not token_set or not can_partition(s, blocks):
        tokens_bin.append(s)
        token_set.add(s)
        token_set.add(get_barre(s))

print(f"Tokens irréductibles trouvés ({len(tokens_bin)}) : {[int(t,2) for t in tokens_bin]}")

# Construction du vocabulaire
vocab = ["<PAD>", "<START>", "<END>"]
token_to_bits = {}

# Associer chaque token binaire à son nom normal et barré
bit_to_tokname = {}
for t_bin in tokens_bin:
    val = int(t_bin, 2)
    t_normal = f"T_{val}"
    t_barre = f"B_{val}"
    vocab.append(t_normal)
    vocab.append(t_barre)
    token_to_bits[t_normal] = t_bin
    token_to_bits[t_barre] = get_barre(t_bin)
    bit_to_tokname[t_bin] = t_normal
    bit_to_tokname[get_barre(t_bin)] = t_barre

token2idx = {tok: idx for idx, tok in enumerate(vocab)}
idx2token = {idx: tok for idx, tok in enumerate(vocab)}

def get_bit_representation(tok_str):
    if tok_str in ["<PAD>", "<START>", "<END>"]:
        return ""
    return token_to_bits.get(tok_str, "")

def decode_sequence(tok_seq):
    bit_str = "".join(get_bit_representation(tok) for tok in tok_seq)
    if not bit_str:
        return 0
    return int(bit_str, 2)

# =====================================================================
# 2. ENCODEUR EXHAUSTIF (Toutes les combinaisons possibles)
# =====================================================================

def get_all_partitions(sub_s, available_blocks):
    """Retourne la liste de TOUTES les séquences de tokens possibles pour une chaîne binaire."""
    if not sub_s:
        return [[]]
    
    results = []
    for b_val, tok_name in available_blocks:
        if sub_s.startswith(b_val):
            sub_results = get_all_partitions(sub_s[len(b_val):], available_blocks)
            for sr in sub_results:
                results.append([tok_name] + sr)
    return results

# Préparer la liste de tous les blocs disponibles (normal + barré)
all_available_blocks = []
for t_b in tokens_bin:
    all_available_blocks.append((t_b, bit_to_tokname[t_b]))
    all_available_blocks.append((get_barre(t_b), bit_to_tokname[get_barre(t_b)]))
# Tri par taille décroissante pour l'efficacité récursive
all_available_blocks = sorted(all_available_blocks, key=lambda x: len(x[0]), reverse=True)

print("Génération exhaustive de toutes les décompositions pour le dataset...")
encoded_dataset = []
for p in primes:
    s = bin(p)[2:]
    partitions = get_all_partitions(s, all_available_blocks)
    if partitions:
        for part in partitions:
            # On ajoute balises start/end et on limite la taille excessive si besoin
            if len(part) <= 12:
                encoded_dataset.append(["<START>"] + part + ["<END>"])

print(f"Dataset enrichi prêt : {len(encoded_dataset)} séquences d'entraînement générées.")

pad_idx = token2idx["<PAD>"]
data_tensors = [torch.tensor([token2idx[t] for t in seq], dtype=torch.long) for seq in encoded_dataset]

from torch.nn.utils.rnn import pad_sequence
padded_data = pad_sequence(data_tensors, batch_first=True, padding_value=pad_idx)

inputs = padded_data[:, :-1]
targets = padded_data[:, 1:]

# =====================================================================
# 3. ARCHITECTURE DU TRANSFORMER
# =====================================================================

class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3, max_len=30):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pos_embedding.num_embeddings:
            device = x.device
            self.pos_embedding = nn.Embedding(seq_len + 20, self.pos_embedding.embedding_dim).to(device)

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        out = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc_out(out), None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TokenTransformer(len(vocab)).to(device)

MODEL_FILE = "transformer_exhaustif.pth"

if os.path.exists(MODEL_FILE):
    print("📂 Poids existants trouvés. Chargement...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
else:
    print("🆕 Entraînement du Transformer sur toutes les combinaisons (30 époques)...")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    inputs, targets = inputs.to(device), targets.to(device)
    epochs = 30
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Époque {epoch:02d}/{epochs} | Loss : {loss.item():.4f}")
    
    torch.save(model.state_dict(), MODEL_FILE)
    print("💾 Modèle sauvegardé.")

# =====================================================================
# 4. GÉNÉRATION AVEC RUSTINE ANTI-PARITÉ
# =====================================================================

def generate_with_parity_filter(model, max_tokens=10, temperature=0.8):
    model.eval()
    start_idx = token2idx["<START>"]
    end_idx = token2idx["<END>"]
    pad_idx = token2idx["<PAD>"]
    
    generated = [start_idx]
    current_bit_str = ""

    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1, :] / temperature
            
            next_logits[start_idx] = -float("Inf")
            next_logits[pad_idx] = -float("Inf")
            
            if len(current_bit_str) > 0 and current_bit_str[-1] == "0":
                next_logits[end_idx] = -float("Inf")
            
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == end_idx:
                break
                
            generated.append(next_idx)
            tok_name = idx2token[next_idx]
            current_bit_str += get_bit_representation(tok_name)

    seq_tokens = [idx2token[idx] for idx in generated[1:] if idx not in [end_idx, pad_idx, start_idx]]
    val = decode_sequence(seq_tokens)
    return val, seq_tokens

print("\n--- Génération avec le modèle entraîné sur toutes les combinaisons ---")
valid_primes = 0
total = 50

for _ in range(total):
    val, seq = generate_with_parity_filter(model)
    is_p = isprime(val) if val > 1 else False
    if is_p:
        valid_primes += 1
        tag = "✅ VRAI"
    else:
        tag = "❌ FAUX"
    print(f"Généré : {val:8d} | Séquence: {seq} | Premier ? -> {tag}")

rate = (valid_primes / total) * 100
print(f"\nTaux de réussite : {valid_primes}/{total} ({rate:.1f}%)")