import matplotlib.pyplot as plt
import numpy as np


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

# Dictionnaire pour retrouver la valeur de chaque brique (addition pure)
token_values = {}
for val, norm, barre in BASE_TOKENS:
  token_values[norm] = val
  token_values[barre] = val

all_tokens_sorted = sorted(list(token_values.keys()), key=len, reverse=True)


def find_first_valid_path(sub_s, blks):
  memo = {}

  def backtrack(s):
    if not s:
      return []
    if s in memo:
      return memo[s]
    for b in blks:
      if b and s.startswith(b):
        res = backtrack(s[len(b) :])
        if res is not None:
          path = [b] + res
          memo[s] = path
          return path
    memo[s] = None
    return None

  return backtrack(sub_s)


# 1. Récupérer les nombres premiers jusqu'à 100
primes = [p for p in range(2, 2001) if is_prime(p)]

x_coords = []
y_coords = []
labels = []

# 2. Calculer S (addition pure) et les coordonnées pour chaque nombre premier
for n in primes:
  s_bin = bin(n)[2:]
  path = find_first_valid_path(s_bin, all_tokens_sorted)

  if path is not None:
    S = 0
    for t in path:
      if t in token_values:
        # Addition systématique (plus de soustraction)
        S += token_values[t]

    # Transformation géométrique
    x = n * np.cos(S)
    y = n * np.sin(S)

    x_coords.append(x)
    y_coords.append(y)
    labels.append(n)

# 3. Tracé du graphique
plt.figure(figsize=(9, 9))
scatter = plt.scatter(
    x_coords, y_coords, c=labels, cmap="plasma", s=120, edgecolors="black"
)
plt.colorbar(scatter, label="Valeur du nombre premier (n)")

# Ajout des étiquettes pour chaque nombre premier tracé
for i, txt in enumerate(labels):
  plt.annotate(
      f"{txt}",
      (x_coords[i], y_coords[i]),
      fontsize=9,
      xytext=(4, 4),
      textcoords="offset points",
      alpha=0.8,
  )

plt.axhline(0, color="grey", linewidth=0.5, linestyle="--")
plt.axvline(0, color="grey", linewidth=0.5, linestyle="--")
plt.title(
    "Visualisation géométrique des nombres premiers jusqu'à 100\n(Addition pure"
    " pour S: x = n * cos(S), y = n * sin(S))",
    fontsize=12,
)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()