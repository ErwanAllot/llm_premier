import matplotlib.pyplot as plt

BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

# Récupération de tous les blocs (tokens + barrés)
all_blocks = []
for _, token, barre in BASE_TOKENS:
    all_blocks.append(token)
    all_blocks.append(barre)
# Supprimer les doublons éventuels de blocs
all_blocks = list(set(all_blocks))

def get_all_partitions(sub_s, blks, memo=None):
    """Renvoie la liste de TOUTES les combinaisons uniques possibles pour une chaîne binaire."""
    if memo is None:
        memo = {}
    if not sub_s:
        return [()]  # Un tuple vide représente une solution complète
    if sub_s in memo:
        return memo[sub_s]
    
    paths = set()
    for b in blks:
        if b and sub_s.startswith(b):
            sub_paths = get_all_partitions(sub_s[len(b):], blks, memo)
            for sp in sub_paths:
                # On stocke sous forme de tuple pour pouvoir utiliser un set (éviter les doublons stricts)
                paths.add((b,) + sp)
                
    memo[sub_s] = paths
    return paths

# Paramètres de l'étude
max_test = 5000  # Vous pouvez ajuster cette valeur selon vos envies
x_vals = []
y_vals = []

print("Calcul des combinaisons en cours...")
for n in range(1, max_test + 1):
    s = bin(n)[2:]
    partitions = get_all_partitions(s, all_blocks)
    nb_combinations = len(partitions)
    
    x_vals.append(n)
    y_vals.append(nb_combinations)

# Création du graphique
plt.figure(figsize=(12, 6))
plt.plot(x_vals, y_vals, marker='.', linestyle='-', color='b', alpha=0.7, markersize=4)
plt.title("Nombre de combinaisons uniques pour chaque entier naturel (Système des Vrais Tokens + Barrés)", fontsize=12)
plt.xlabel("Nombre étudié (x)", fontsize=10)
plt.ylabel("Nombre de combinaisons uniques (y)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.yscale('log') # Échelle logarithmique souvent très utile ici car le nombre de combinaisons explose vite
plt.tight_layout()

print("Affichage du graphique...")
plt.show()