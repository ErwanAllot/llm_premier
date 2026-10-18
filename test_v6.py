import os
import math
import random
import torch
import torch.nn as nn
from sympy import isprime, primerange

# =====================================================================
# 1. PRÉPARATION DU VOCABULAIRE & TOKENISATION
# =====================================================================

# Extraction d'une grande base de premiers pour fixer les tokens
N_PRIMES_TRAIN = 10000
primes_list = list(primerange(2, 105000))[:N_PRIMES_TRAIN]

# Extraction des blocs binaires (morceaux de 3 à 5 bits) issus des nombres premiers
prime_blocks = set()
for p in primes_list:
    b = bin(p)[2:]
    i = 0
    while i < len(b):
        chunk_len = min(random.randint(3, 5), len(b) - i)
        val = int(b[i:i+chunk_len], 2)
        if val > 1:
            prime_blocks.add(val)
        i += chunk_len

prime_blocks = sorted(list(prime_blocks))

# Construction du vocabulaire fixe
vocab = ["<PAD>", "<START>", "<END>"]
for p in prime_blocks:
    vocab.append(f"T_{p}")
    vocab.append(f"B_{p}")  # Barré / inversé

token2idx = {tok: idx for idx, tok in enumerate(vocab)}
idx2token = {idx: tok for idx, tok in enumerate(vocab)}

def get_bit_representation(tok_str):
    """Retourne la chaîne de bits associée à un token (T_x ou B_x)."""
    if tok_str in ["<PAD>", "<START>", "<END>"]:
        return ""
    tag, val_str = tok_str.split("_")
    val = int(val_str)
    b = bin(val)[2:]
    if tag == "B":
        b = "".join("1" if char == "0" else "0" for char in b)
    return b

def encode_prime(p):
    """Encode un nombre premier p en séquence de tokens."""
    b = bin(p)[2:]
    seq = []
    i = 0
    while i < len(b):
        matched = False
        for sz in range(5, 2, -1):
            if i + sz <= len(b):
                chunk = b[i:i+sz]
                val = int(chunk, 2)
                if val in prime_blocks:
                    seq.append(f"T_{val}")
                    i += sz
                    matched = True
                    break
                inv_chunk = "".join("1" if c == "0" else "0" for c in chunk)
                inv_val = int(inv_chunk, 2)
                if inv_val in prime_blocks:
                    seq.append(f"B_{inv_val}")
                    i += sz
                    matched = True
                    break
        if not matched:
            bit = b[i]
            seq.append("T_3" if bit == "1" else "B_3")
            i += 1
    return ["<START>"] + seq + ["<END>"]

def decode_sequence(tok_seq):
    """Reconstitue la chaîne binaire puis l'entier décimal."""
    bit_str = "".join(get_bit_representation(tok) for tok in tok_seq)
    if not bit_str:
        return 0
    return int(bit_str, 2)

# =====================================================================
# 2. ARCHITECTURE DU TRANSFORMER (MINI-GPT)
# =====================================================================

class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=3, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        
        # Sécurité dynamique pour éviter le dépassement de position
        if seq_len > self.pos_embedding.num_embeddings:
            device = x.device
            self.pos_embedding = nn.Embedding(seq_len + 20, self.pos_embedding.embedding_dim).to(device)

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        out = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        out = self.transformer(out, mask=mask, is_causal=True)
        logits = self.fc_out(out)
        return logits, None

# =====================================================================
# 3. PRÉPARATION DU DATASET ET ENTRAÎNEMENT
# =====================================================================

encoded_dataset = [encode_prime(p) for p in primes_list]
max_seq_len = max(len(s) for s in encoded_dataset)

pad_idx = token2idx["<PAD>"]
data_tensors = []
for seq in encoded_dataset:
    indices = [token2idx[t] for t in seq]
    data_tensors.append(torch.tensor(indices, dtype=torch.long))

from torch.nn.utils.rnn import pad_sequence
padded_data = pad_sequence(data_tensors, batch_first=True, padding_value=pad_idx)

inputs = padded_data[:, :-1]
targets = padded_data[:, 1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TokenTransformer(len(vocab), max_len=max_seq_len+5).to(device)

MODEL_FILE = "transformer_premiers.pth"

if os.path.exists(MODEL_FILE):
    print("📂 Fichier 'transformer_premiers.pth' trouvé. Chargement des poids...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
else:
    print("🆕 Aucun poids détecté. Entraînement initial du Transformer...")

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
inputs = inputs.to(device)
targets = targets.to(device)

epochs = 15
print(f"Lancement de {epochs} époques d'apprentissage...")
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    print(f"Époque {epoch:02d}/{epochs} | Pertes (Loss) : {loss.item():.4f}")

torch.save(model.state_dict(), MODEL_FILE)
print(f"💾 Modèle Transformer sauvegardé dans '{MODEL_FILE}'.\n")

# =====================================================================
# 4. GÉNÉRATION AVEC RUSTINES DE STRUCTURE & ANTI-PARITÉ
# =====================================================================

def generate_with_parity_filter(model, max_tokens=15, temperature=0.9):
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
            
            # --- INTERDICTIONS STRICTES (RUSTINES DE STRUCTURE) ---
            # Interdiction absolue de générer <START> ou <PAD> en cours de route
            next_logits[start_idx] = -float("Inf")
            next_logits[pad_idx] = -float("Inf")
            
            # --- RUSTINE ANTI-PARITÉ ---
            # Si le nombre en cours se termine par '0' (pair), on interdit de s'arrêter (<END>)
            if len(current_bit_str) > 0 and current_bit_str[-1] == "0":
                next_logits[end_idx] = -float("Inf")
            
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == end_idx:
                break
                
            generated.append(next_idx)
            tok_name = idx2token[next_idx]
            current_bit_str += get_bit_representation(tok_name)

    # Nettoyage final de sécurité
    seq_tokens = [idx2token[idx] for idx in generated[1:] if idx not in [end_idx, pad_idx, start_idx]]
    val = decode_sequence(seq_tokens)
    return val, seq_tokens

# Test de génération sur 50 nombres
print("--- Génération de nombres par le Transformer (+ Rustines) ---")
valid_primes = 0
total_generated = 50

for _ in range(total_generated):
    val, seq = generate_with_parity_filter(model)
    is_p = isprime(val) if val > 1 else False
    if is_p:
        valid_primes += 1
        tag = "✅ VRAI"
    else:
        tag = "❌ FAUX (Composé)"
    print(f"Généré : {val:8d} | Séquence: {seq} | Premier ? -> {tag}")

rate = (valid_primes / total_generated) * 100
print(f"\nRésultat : {valid_primes} / {total_generated} nombres générés sont des VRAIS premiers ({rate:.1f}%) !")