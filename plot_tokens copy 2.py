import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


def is_prime(num):
  if num < 2:
    return False
  for i in range(2, int(num**0.5) + 1):
    if num % i == 0:
      return False
  return True


# BASE_TOKENS : (valeur_decimale, token_normal, token_barre)
BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110"),
]

# Dictionnaires d'association pour savoir si une brique est normale ou barre, et sa valeur
token_info = {}
for val, norm, barre in BASE_TOKENS:
  token_info[norm] = ("prime", val)
  token_info[barre] = ("barre", val)

# Notre alphabet de briques + un token spécial de fin '<EOS>'
vocab_tokens = sorted(list(token_info.keys()))
vocab_tokens.append("<EOS>")
vocab_size = len(vocab_tokens)

token_to_idx = {tok: i for i, tok in enumerate(vocab_tokens)}
idx_to_token = {i: tok for i, tok in enumerate(vocab_tokens)}
eos_idx = token_to_idx["<EOS>"]


def find_first_valid_path(sub_s, blks):
  sorted_blks = sorted(list(set(blks)), key=len, reverse=True)
  memo = {}

  def backtrack(s):
    if not s:
      return []
    if s in memo:
      return memo[s]
    for b in sorted_blks:
      if b and s.startswith(b):
        res = backtrack(s[len(b) :])
        if res is not None:
          path = [b] + res
          memo[s] = path
          return path
    memo[s] = None
    return None

  return backtrack(sub_s)


# --- CORPUS STRUCTURÉ AVEC DÉLIMITATEUR ---
print("Construction du corpus structuré avec <EOS>...")
corpus_sequences = []
for p in range(2, 100000):
  if is_prime(p):
    s = bin(p)[2:]
    real_tokens = [t for t in vocab_tokens if t != "<EOS>"]
    path = find_first_valid_path(s, real_tokens)
    if path is not None:
      corpus_sequences.append([token_to_idx[t] for t in path] + [eos_idx])

data = []
for seq in corpus_sequences:
  data.extend(seq)
data = torch.tensor(data, dtype=torch.long)
print(f"Taille du corpus structuré : {len(data)} tokens.")


# --- ARCHITECTURE DU TRANSFORMER ---
class TokenTransformer(nn.Module):

  def __init__(
      self, vocab_size, d_model=128, n_heads=8, n_layers=4, block_size=16
  ):
    super().__init__()
    self.block_size = block_size
    self.token_embedding = nn.Embedding(vocab_size, d_model)
    self.pos_embedding = nn.Embedding(block_size, d_model)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        batch_first=True,
    )
    self.transformer = nn.TransformerEncoder(
        encoder_layer, num_layers=n_layers
    )
    self.fc_out = nn.Linear(d_model, vocab_size)

  def forward(self, idx):
    B, T = idx.shape
    device = idx.device
    pos = torch.arange(0, T, device=device).unsqueeze(0)

    tok_emb = self.token_embedding(idx)
    pos_emb = self.pos_embedding(pos)
    x = tok_emb + pos_emb

    # Masque causal pour éviter de regarder le futur
    mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
    x = self.transformer(x, mask=mask, is_causal=True)
    logits = self.fc_out(x)
    return logits

  @torch.no_grad()
  def generate(self, max_new_tokens=12, temperature=0.8, top_k=5):
    start_t = torch.randint(
        0, vocab_size - 1, (1, 1)
    )  # On évite de commencer par EOS
    idx = start_t

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size :]
      logits = self(idx_cond)[:, -1, :]
      logits = logits / temperature

      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("Inf")

      probs = F.softmax(logits, dim=-1)
      next_idx = torch.multinomial(probs, num_samples=1)

      if next_idx.item() == eos_idx:
        break

      idx = torch.cat((idx, next_idx), dim=1)
    return idx


# --- ENTRAÎNEMENT ---
block_size = 16
batch_size = 64
model = TokenTransformer(vocab_size, block_size=block_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

print("Entraînement du modèle structuré...")
model.train()
for epoch in range(400):
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i : i + block_size] for i in ix])
  y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

  optimizer.zero_grad()
  loss = F.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
  loss.backward()
  optimizer.step()

print("Entraînement terminé ! Génération et analyse géométrique...")

# --- GÉNÉRATION ET VISUALISATION POLAIRE ---
generated_primes_data = []

for _ in range(100):  # On génère 100 échantillons pour le graphique
  gen_tokens_idx = model.generate(max_new_tokens=10).squeeze().tolist()
  if isinstance(gen_tokens_idx, int):
    gen_tokens_idx = [gen_tokens_idx]

  # Nettoyage jusqu'au premier EOS s'il y en a un
  clean_tokens = []
  for idx in gen_tokens_idx:
    if idx == eos_idx:
      break
    clean_tokens.append(idx)

  if not clean_tokens:
    continue

  token_strs = [idx_to_token[i] for i in clean_tokens]

  # Reconstitution du binaire et calcul décimal
  bin_str = "".join(token_strs)
  try:
    dec_val = int(bin_str, 2)
  except ValueError:
    continue

  if is_prime(dec_val):
    generated_primes_data.append((dec_val, token_strs))

# --- TRACÉ DU GRAPHIQUE ---
x_coords = []
y_coords = []
primes_labels = []

for n, t_list in generated_primes_data:
  S = 0
  for t in t_list:
    if t in token_info:
      t_type, val = token_info[t]
      if t_type == "prime":
        S += val
      elif t_type == "barre":
        S -= val

  # Transformation géométrique demandée
  x = n * np.cos(S)
  y = n * np.sin(S)

  x_coords.append(x)
  y_coords.append(y)
  primes_labels.append(n)

plt.figure(figsize=(9, 9))
scatter = plt.scatter(
    x_coords,
    y_coords,
    c=primes_labels,
    cmap="plasma",
    s=120,
    edgecolors="black",
)
plt.colorbar(scatter, label="Valeur décimale (n)")

# Annotation des points générés
for i, txt in enumerate(primes_labels):
  plt.annotate(
      f"n={txt}",
      (x_coords[i], y_coords[i]),
      fontsize=8,
      xytext=(4, 4),
      textcoords="offset points",
      alpha=0.8,
  )

plt.axhline(0, color="grey", linewidth=0.5, linestyle="--")
plt.axvline(0, color="grey", linewidth=0.5, linestyle="--")
plt.title(
    "Trajectoire Géométrique des Nombres Premiers Générés\n(x = n * cos(S), y ="
    " n * sin(S))",
    fontsize=12,
)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()