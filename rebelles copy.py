# les rebelles
# 4 : 100
# 6 : 110
# 25 : 11001
# 27 : 11011
# 100 : 1100100
# 102 : 1100110
# 411 : 110011011


# Définition de vos 7 tokens de base avec leurs binaires et leurs barrés
BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

# Construire la liste de tous les blocs autorisés (tokens + barrés)
blocks = []
for val, b_str, barre_str in BASE_TOKENS:
    blocks.append(b_str)
    blocks.append(barre_str)

def can_partition(sub_s, blks):
    """Vérifie si une chaîne binaire peut être entièrement découpée par les blocs."""
    if not sub_s:
        return True
    # Trier par longueur décroissante (priorité aux plus grands blocs)
    unique_blks = sorted(list(set(blks)), key=len, reverse=True)
    for b in unique_blks:
        if b and sub_s.startswith(b):
            if can_partition(sub_s[len(b):], blks):
                return True
    return False

# Recherche des nombres naturels non partitionnables après 411
print("Recherche des nombres naturels non partitionnables par l'alphabet...")
non_partitionables = []

# On teste les nombres à partir de 412 jusqu'à une limite de test
limite = 1000000
for n in range(412, limite + 1):
    s = bin(n)[2:]
    if not can_partition(s, blocks):
        non_partitionables.append(n)

print(f"Nombres non partitionnables trouvés entre 412 et {limite} :")
print(non_partitionables)