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
all_blocks = list(set(all_blocks))

def is_prime(num):
    """Vérifie si un nombre est premier."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def get_all_partitions(sub_s, blks, memo=None):
    """Renvoie la liste de TOUTES les combinaisons uniques possibles pour une chaîne binaire."""
    if memo is None:
        memo = {}
    if not sub_s:
        return [()]
    if sub_s in memo:
        return memo[sub_s]
    
    paths = set()
    for b in blks:
        if b and sub_s.startswith(b):
            sub_paths = get_all_partitions(sub_s[len(b):], blks, memo)
            for sp in sub_paths:
                paths.add((b,) + sp)
                
    memo[sub_s] = paths
    return paths

# Paramètres de l'étude
max_test = 50
x_compose, y_compose = [], []
x_premiers, y_premiers = [], []

print("Calcul des combinaisons et tri des nombres premiers en cours...")
for n in range(1, max_test + 1):
    s = bin(n)[2:]
    partitions = get_all_partitions(s, all_blocks)
    nb_combinations = len(partitions)
    
    if is_prime(n):
        x_premiers.append(n)
        y_premiers.append(nb_combinations)
    else:
        x_compose.append(n)
        y_compose.append(nb_combinations)

# Création du graphique
plt.figure(figsize=(12, 6))

# Tracé des nombres composés en bleu
plt.scatter(x_compose, y_compose, color='blue', s=15, alpha=0.6, label='Composés')

# Tracé des nombres premiers en ROUGE (avec des points plus gros pour les repérer)
plt.scatter(x_premiers, y_premiers, color='red', s=35, zorder=5, label='Premiers 🌟')

plt.title("Nombre de combinaisons uniques (Bleu = Composés, Rouge = Premiers)", fontsize=12)
plt.xlabel("Nombre étudié (x)", fontsize=10)
plt.ylabel("Nombre de combinaisons uniques (y)", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)
plt.yscale('log')
plt.legend()
plt.tight_layout()

print("Affichage du graphique...")
plt.show()