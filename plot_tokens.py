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

# Dictionnaire pour retrouver le type et la valeur de chaque brique
token_info = {}
for val, norm, barre in BASE_TOKENS:
  token_info[norm] = ("prime", val)
  token_info[barre] = ("barre", val)

all_tokens_sorted = sorted(list(token_info.keys()), key=len, reverse=True)


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


# 1. Récupérer les 100 premiers nombres premiers
primes = []
candidate = 2
while len(primes) < 1000:
  if is_prime(candidate):
    primes.append(candidate)
  candidate += 1

x_coords = []
y_coords = []
labels = []

# 2. Calculer S et les coordonnées (x, y) pour chaque nombre premier
for n in primes:
  s_bin = bin(n)[2:]
  path = find_first_valid_path(s_bin, all_tokens_sorted)

  if path is not None:
    # Calcul de S selon vos règles (+ si prime, - si barre)
    S = 0
    for t in path:
      if t in token_info:
        t_type, val = token_info[t]
        if t_type == "prime":
          S += val
        elif t_type == "barre":
          S -= val

    # Transformation géométrique
    x = n * np.cos(S)
    y = n * np.sin(S)

    x_coords.append(x)
    y_coords.append(y)
    labels.append(n)

# 3. Tracé du graphique
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x_coords, y_coords, c=labels, cmap="plasma", s=100, edgecolors="black")
plt.colorbar(scatter, label="Valeur du nombre premier (n)")

# Ajout des étiquettes des nombres
for i, txt in enumerate(labels):
  plt.annotate(f"{txt}", (x_coords[i], y_coords[i]), fontsize=8, xytext=(4, 4), textcoords="offset points", alpha=0.7)

plt.axhline(0, color="grey", linewidth=0.5, linestyle="--")
plt.axvline(0, color="grey", linewidth=0.5, linestyle="--")
plt.title("Visualisation géométrique des 100 premiers nombres premiers\n(x = n * cos(S), y = n * sin(S))", fontsize=12)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()