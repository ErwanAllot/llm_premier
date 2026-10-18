import matplotlib.pyplot as plt
import numpy as np

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

# Dictionnaire pour retrouver directement la valeur de chaque brique (qu'elle soit normale ou barre)
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


# Paramètre : jusqu'à quel nombre naturel on veut tracer
max_n = 2500

x_coords = []
y_coords = []
labels = []

# Parcourir tous les nombres naturels
for n in range(2, max_n + 1):
  s_bin = bin(n)[2:]
  path = find_first_valid_path(s_bin, all_tokens_sorted)

  # Si le nombre peut être entièrement découpé avec nos briques
  if path is not None:
    S = 0
    for t in path:
      if t in token_values:
        # Addition systématique de la valeur de la brique (plus de soustraction)
        S += token_values[t]

    # Transformation géométrique
    x = n * np.cos(S)
    y = n * np.sin(S)

    x_coords.append(x)
    y_coords.append(y)
    labels.append(n)

# Tracé du graphique
plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    x_coords, y_coords, c=labels, cmap="viridis", s=60, edgecolors="black", alpha=0.8
)
plt.colorbar(scatter, label="Valeur de n")

# Pour éviter d'encombrer le graphique, on n'affiche les étiquettes que s'il y a peu de points
if len(labels) <= 100:
  for i, txt in enumerate(labels):
    plt.annotate(
        f"{txt}",
        (x_coords[i], y_coords[i]),
        fontsize=7,
        xytext=(3, 3),
        textcoords="offset points",
        alpha=0.7,
    )

plt.axhline(0, color="grey", linewidth=0.5, linestyle="--")
plt.axvline(0, color="grey", linewidth=0.5, linestyle="--")
plt.title(
    f"Visualisation géométrique avec addition pure pour S (2 à {max_n})\n(x ="
    " n * cos(S), y = n * sin(S))",
    fontsize=12,
)
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()