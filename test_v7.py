import os
import torch
import torch.nn as nn
from sympy import isprime

# =====================================================================
# 1. VOTRE VOCABULAIRE STRICT DE 7 TOKENS (Premiers + Barrés)
# =====================================================================

# Définition exacte de vos 7 tokens de base et de leurs représentations binaires
BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

# Construction du vocabulaire
vocab = ["<PAD>", "<START>", "<END>"]
token_to_bits = {}

for val, b_normal, b_barre in BASE_TOKENS:
    t_normal = f"T_{val}"
    t_barre = f"B_{val}"
    vocab.append(t_normal)
    vocab.append(t_barre)
    token_to_bits[t_normal] = b_normal
    token_to_bits[t_barre] = b_barre

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
# 2. GÉNÉRATION D'UN PETIT DATASET D'EXEMPLES POUR TESTER
# =====================================================================
# Pour que le modèle puisse s'entraîner, on reconstruit quelques séquences valides
# basées sur la concaténation de vos tokens (ou on prend de vrais premiers formés ainsi)

# Exemple de séquences encodées à partir de vos tokens
sample_sequences = [
    ["<START>", "T_3", "T_5", "<END>"],          # 3 et 5 -> 35 (composé mais pour l'exemple)
    ["<START>", "T_2", "T_7", "<END>"],          # 2 et 7 -> 27 (composé)
    ["<START>", "T_17", "T_3", "<END>"],         # 17 et 3 -> 173 (premier !)
    ["<START>", "B_3", "T_19", "<END>"],
    ["<START>", "T_5", "T_3", "<END>"],          # 5 et 3 -> 53 (premier !)
    ["<START>", "T_7", "B_5", "<END>"],
    ["<START>", "T_2", "T_17", "<END>"],
    ["<START>", "T_19", "T_3", "<END>"],         # 19 et 3 -> 193 (premier !)
]

# On transforme ça en tenseurs
pad_idx = token2idx["<PAD>"]
data_tensors = [torch.tensor([token2idx[t] for t in seq], dtype=torch.long) for seq in sample_sequences]

from torch.nn.utils.rnn import pad_sequence
padded_data = pad_sequence(data_tensors, batch_first=True, padding_value=pad_idx)

inputs = padded_data[:, :-1]
targets = padded_data[:, 1:]

# =====================================================================
# 3. ARCHITECTURE DU TRANSFORMER
# =====================================================================

class TokenTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2, hidden_dim=64, num_layers=2, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        out = self.embedding(x) + self.pos_embedding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc_out(out), None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TokenTransformer(len(vocab)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entraînement rapide sur vos exemples
model.train()
inputs, targets = inputs.to(device), targets.to(device)
for epoch in range(50):
    optimizer.zero_grad()
    logits, _ = model(inputs)
    loss = criterion(logits.reshape(-1, len(vocab)), targets.reshape(-1))
    loss.backward()
    optimizer.step()

print("✅ Modèle initialisé et entraîné sur vos 7 tokens de base.\n")

# =====================================================================
# 4. GÉNÉRATION AVEC RUSTINE ANTI-PARITÉ
# =====================================================================

def generate_with_parity_filter(model, max_tokens=6, temperature=0.8):
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

# Test
print("--- Génération avec vos 7 tokens stricts ---")
for _ in range(10):
    val, seq = generate_with_parity_filter(model)
    is_p = isprime(val) if val > 1 else False
    tag = "✅ VRAI" if is_p else "❌ FAUX"
    print(f"Généré : {val:6d} | Séquence: {seq} | Premier ? -> {tag}")