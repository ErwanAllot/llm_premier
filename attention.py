def get_barre(b_str):
    """Génère le bitwise barré d'une chaîne binaire."""
    return "".join('1' if c == '0' else '0' for c in b_str)

def is_prime(num):
    """Vérifie si un nombre est premier."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Les rebelles et leurs représentations binaires
rebelles_dict = {
    4: "100",
    6: "110",
    25: "11001",
    27: "11011",
    100: "1100100",
    102: "1100110",
    411: "110011011"
}

rebelles_tokens = list(rebelles_dict.values())
rebelles_tokens_barre = [get_barre(b) for b in rebelles_tokens]
all_rebel_blocks = rebelles_tokens + rebelles_tokens_barre

def get_partition_path(sub_s, blks, memo=None):
    """Renvoie la liste des blocs utilisés si constructible, sinon None."""
    if memo is None:
        memo = {}
    if not sub_s:
        return []
    if sub_s in memo:
        return memo[sub_s]
    
    unique_blks = sorted(list(set(blks)), key=len, reverse=True)
    for b in unique_blks:
        if b and sub_s.startswith(b):
            res = get_partition_path(sub_s[len(b):], blks, memo)
            if res is not None:
                path = [b] + res
                memo[sub_s] = path
                return path
                
    memo[sub_s] = None
    return None

# Testons les nombres naturels jusqu'à 500 pour voir les quelques élus
max_test = 10000
print(f"=== SUCCÈS DU CLAN DES REBELLES (1 à {max_test}) ===")

succes_count = 0
for n in range(1, max_test + 1):
    s = bin(n)[2:]
    path = get_partition_path(s, all_rebel_blocks)
    
    if path is not None:
        succes_count += 1
        prime_statuspen = "EST PREMIER 🌟" if is_prime(n) else "composé"
        print(f"Nombre {n:3d} (binaire: {s:>10}) -> Découpage : {path} | {prime_statuspen}")

print(f"\nTotal des nombres réussis sur {max_test} : {succes_count}")