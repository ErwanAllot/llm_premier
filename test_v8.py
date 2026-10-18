import os
import torch
import torch.nn as nn
from sympy import isprime, primerange

# =====================================================================
# 1. DÉFINITION DE VOTRE VOCABULAIRE STRICT DE 7 TOKENS
# =====================================================================

BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

vocab = ["<PAD>", "<START>", "<END>"]
token_to_bits = {}
allowed_vals = []

for val, b_normal, b_barre in BASE_TOKENS:
    t_normal = f"T_{val}"
    t_barre = f"B_{val}"
    vocab.append(t_normal)
    vocab.append(t_barre)
    token_to_bits[t_normal] = b_normal
    token_to_bits[t_barre] = b_barre
    allowed_vals.append(val)

# Tri des valeurs par taille binaire décroissante pour un découpage glouton optimal
allowed_vals = sorted(allowed_vals, key=lambda x: len(bin(x)[2:]), reverse=True)

token2idx = {tok: idx for idx, tok in enumerate(vocab)}
idx2token = {idx: tok for idx, tok in enumerate(vocab)}

def get_bit_representation(tok_str):
    if tok_str in ["<PAD>", "<START>", "<END>"]:
        return ""
    return token_to_bits.get(tok_str, "")

def encode_prime_strict(p):
    """Encode un nombre premier en utilisant STRICTEMENT vos 7 tokens de base."""
    b = bin(p)[2:]
    seq = []
    i = 0
    while i < len(b):
        matched = False
        for val in allowed_vals:
            # Test normal
            b_norm = bin(val)[2:]
            if b.startswith(b_norm, i):
                seq.append(f"T_{val}")
                i += len(b_norm)
                matched = True
                break
            # Test barré
            b_bar = "".join("1" if c == "0" else "0" for c in b_norm)
            if b.startswith(b_bar, i):
                seq.append(f"B_{val}")
                i += len(b_bar)
                matched = True
                break
        if not matched:
            # Si le motif ne correspond pas exactement, on ignore ce nombre pour le dataset d'entraînement
            return None
        if len(seq) > 12:  # Éviter les séquences trop longues
            return None
    return ["<START>"] + seq + ["<END>"]

def decode_sequence(tok_seq):
    bit_str = "".join(get_bit_representation(tok) for tok in tok_seq)
    if not bit_str:
        return 0
    return int(bit_str, 2)

# =====================================================================
# 2. PRÉPARATION DU DATASET SUR 10 000 PREMIERS
# =====================================================================

print("Extraction et encodage des 10 000 premiers nombres premiers...")
N_PRIMES = 10000
primes_list = list(primerange(2, 120000))[:N_PRIMES]

encoded_dataset = []
for p in primes_list:
    res = encode_prime_strict(p)
    if res is not None:
        encoded_dataset.append(res)

print(f"Dataset prêt : {len(encoded_dataset)} séquences valides retenues sur {N_PRIMES} testées.")

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

MODEL_FILE = "transformer_strict_7tokens.pth"

if os.path.exists(MODEL_FILE):
    print("📂 Poids existants trouvés. Chargement...")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
else:
    print("🆕 Entraînement du Transformer sur les 10 000 premiers...")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    inputs, targets = inputs.to(device), targets.to(device)
    epochs = 15
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
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
            
            # Interdictions de structure
            next_logits[start_idx] = -float("Inf")
            next_logits[pad_idx] = -float("Inf")
            
            # Rustine anti-parité : si le dernier bit cumulé est '0', on interdit <END>
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

print("\n--- Génération avec le modèle entraîné sur vos 7 tokens stricts ---")
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