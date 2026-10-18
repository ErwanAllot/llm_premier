import os
import random
import torch
import torch.nn as nn

MODEL_PATH = "modele_premiers.pth"

# --- 1. PRÉPARATION DES DONNÉES & TOKENS PREMIERS ---

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def get_barre(s):
    return "".join('1' if c == '0' else '0' for c in s)

# Générer les 1500 premiers nombres premiers
primes = []
n = 2
while len(primes) < 1500:
    if is_prime(n):
        primes.append(n)
    n += 1

# Trouver les tokens irréductibles
tokens_bin = []
token_set = set()

for p in primes:
    s = bin(p)[2:]
    blocks = []
    for t in tokens_bin:
        blocks.append(t)
        blocks.append(get_barre(t))
    
    def can_partition(sub_s, blks):
        if not sub_s: return True
        unique_blks = sorted(list(set(blks)), key=len, reverse=True)
        for b in unique_blks:
            if b and sub_s.startswith(b):
                if can_partition(sub_s[len(b):], blks):
                    return True
        return False

    if not token_set or not can_partition(s, blocks):
        tokens_bin.append(s)
        token_set.add(s)
        token_set.add(get_barre(s))

print(f"Tokens premiers trouvés ({len(tokens_bin)}) : {[int(t,2) for t in tokens_bin]}")

def encode_prime(p, tokens_list):
    s = bin(p)[2:]
    all_blocks = []
    for t in tokens_list:
        all_blocks.append((t, f"T_{int(t,2)}"))
        all_blocks.append((get_barre(t), f"B_{int(t,2)}"))
    
    all_blocks.sort(key=lambda x: len(x[0]), reverse=True)
    
    seq = []
    curr = s
    while curr:
        matched = False
        for b_str, b_label in all_blocks:
            if curr.startswith(b_str):
                seq.append(b_label)
                curr = curr[len(b_str):]
                matched = True
                break
        if not matched:
            break
    return seq

vocab = ["<PAD>", "<START>", "<END>"]
for t in tokens_bin:
    vocab.append(f"T_{int(t,2)}")
    vocab.append(f"B_{int(t,2)}")

token2idx = {tok: i for i, tok in enumerate(vocab)}
idx2token = {i: tok for i, tok in enumerate(vocab)}

encoded_dataset = []
for p in primes:
    seq = ["<START>"] + encode_prime(p, tokens_bin) + ["<END>"]
    encoded_dataset.append([token2idx[t] for t in seq])

# --- 2. LE RÉSEAU DE NEURONES (Mini-LSTM) ---

class TokenRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

model = TokenRNN(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 3. GESTION DE LA SAUVEGARDE / CHARGEMENT ---

if os.path.exists(MODEL_PATH):
    print(f"\n📂 Fichier de sauvegarde détecté ('{MODEL_PATH}'). Chargement des poids...")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("✅ Poids chargés avec succès ! (Pas besoin de réentraîner)")
else:
    print(f"\n--- Début de l'entraînement initial (1500 nombres premiers) ---")
    epochs = 25
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(encoded_dataset)
        for seq in encoded_dataset:
            if len(seq) < 2: continue
            inputs = torch.tensor(seq[:-1]).unsqueeze(0)
            targets = torch.tensor(seq[1:]).unsqueeze(0)

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Époque {epoch+1}/{epochs} | Pertes (Loss) : {total_loss / len(encoded_dataset):.4f}")
    
    # Sauvegarde des poids
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Modèle sauvegardé dans '{MODEL_PATH}'")

# --- 4. GÉNÉRATION DE NOUVEAUX NOMBRES ---

def decode_to_val(token_seq):
    bin_str = ""
    for t_name in token_seq:
        if t_name.startswith("T_"):
            val = int(t_name.split("_")[1])
            bin_str += bin(val)[2:]
        elif t_name.startswith("B_"):
            val = int(t_name.split("_")[1])
            b_str = bin(val)[2:]
            bin_str += get_barre(b_str)
    return int(bin_str, 2) if bin_str else None

print("\n--- Génération de nombres par le Modèle ---")
model.eval()

generated_primes = 0
total_generated = 50
seen_generated = set()

with torch.no_grad():
    for _ in range(total_generated):
        curr_token = torch.tensor([[token2idx["<START>"]]])
        hidden = None
        seq = []
        
        for _ in range(15):
            logits, hidden = model(curr_token, hidden)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            next_tok = idx2token[next_idx]
            if next_tok == "<END>":
                break
            seq.append(next_tok)
            curr_token = torch.tensor([[next_idx]])
            
        val = decode_to_val(seq)
        if val and val not in seen_generated:
            seen_generated.add(val)
            is_p = is_prime(val)
            if is_p: generated_primes += 1
            print(f"Généré : {val:8d} | Séquence: {seq} | Premier ? -> {'✅ VRAI' if is_p else '❌ FAUX (Composé)'}")

print(f"\nRésultat : {generated_primes} / {len(seen_generated)} nombres uniques générés sont des VRAIS premiers ({generated_primes/len(seen_generated)*100:.1f}%) !")