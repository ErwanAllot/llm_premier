import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

# --- 1. CONFIGURATION ET DONNÉES ---
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
all_blocks = list(set([t for _, t, _ in BASE_TOKENS] + [b for _, _, b in BASE_TOKENS]))

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

# Préparation du corpus d'entraînement : les binaires des nombres premiers jusqu'à 2000
max_train_prime = 2000
primes = [n for n in range(2, max_train_prime + 1) if is_prime(n)]
corpus = "".join([bin(p)[2:] + " " for p in primes]) # Espace comme séparateur

chars = sorted(list(set(corpus)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(corpus), dtype=torch.long)

# --- 2. ARCHITECTURE DU MINI-TRANSFORMER (GPT-like) ---
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, block_size=32):
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
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        # Masque causal pour le Transformer
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.fc_out(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=15):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] # Dernier timestep
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# --- 3. ENTRAÎNEMENT ET SAUVEGARDE DU .PTH ---
model_path = "modele_premiers.pth"
block_size = 32
batch_size = 32

model = MiniTransformer(vocab_size, block_size=block_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("Entraînement du mini-Transformer sur les binaires des nombres premiers...")
model.train()
for epoch in range(300): # 300 epochs pour l'exemple
    # Échantillonnage aléatoire de batchs
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()

# Sauvegarde du fichier .pth
torch.save(model.state_dict(), model_path)
print(f"Modèle entraîné et sauvegardé avec succès dans '{model_path}' !")

# --- 4. CHARGEMENT ET TEST DU MODÈLE ---
model.load_state_dict(torch.load(model_path))
model.eval()

print("\n=== GÉNÉRATION ET ÉVALUATION PAR LE TRANSFORMER ===")
num_samples = 50
valid_syntax_count = 0
valid_prime_count = 0

for _ in range(num_samples):
    # Point de départ aléatoire (ex: un '1' ou un début de séquence)
    context = torch.tensor([[stoi['1']]], dtype=torch.long)
    generated = model.generate(context, max_new_tokens=12)
    gen_text = decode(generated[0].tolist()).strip()
    
    # On nettoie pour ne garder que la première partie avant l'espace
    candidate_bin = gen_text.split(" ")[0]
    
    if len(candidate_bin) > 0 and all(c in '01' for c in candidate_bin):
        decimal_val = int(candidate_bin, 2)
        path = find_first_valid_path(candidate_bin, all_blocks)
        
        is_syntax_ok = (path is not None)
        is_p = is_prime(decimal_val) if is_syntax_ok else False
        
        if is_syntax_ok: valid_syntax_count += 1
        if is_p: valid_prime_count += 1
        
        print(f"Généré : {candidate_bin:>10} ( décimal: {decimal_val:5d} ) -> Syntaxique: {'OUI' if is_syntax_ok else 'NON'} | Premier: {'🌟 OUI' if is_p else 'non'}")

print(f"\nTaux de réussite syntaxique du Transformer : {(valid_syntax_count / num_samples) * 100:.1f}%")
print(f"Taux de nombres premiers générés : {(valid_prime_count / num_samples) * 100:.1f}%")