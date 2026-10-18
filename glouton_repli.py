def is_prime(num):
    """Vérifie si un nombre est premier."""
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

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

def find_first_valid_path(sub_s, blks):
    """
    Approche glouton-prioritaire avec repli (backtracking s'arrêtant au 1er succès).
    On teste d'abord les plus grands blocs (glouton), et si ça coince, 
    on explore les autres chemins jusqu'à valider une combinaison.
    """
    sorted_blks = sorted(list(set(blks)), key=len, reverse=True)
    memo = {}
    
    def backtrack(s):
        if not s:
            return []
        if s in memo:
            return memo[s]
        
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

# Étude exclusive des nombres premiers jusqu'à une certaine taille
max_test = 10000
primes = [n for n in range(2, max_test + 1) if is_prime(n)]

succes_count = 0
echec_count = 0

print(f"=== TEST GLOUTON AVEC REPLI SUR LES PREMIERS (1 à {max_test}) ===")
for p in primes:
    s = bin(p)[2:]
    path = find_first_valid_path(s, all_blocks)
    
    if path is not None:
        succes_count += 1
    else:
        echec_count += 1

print(f"\nRésultats pour {len(primes)} nombres premiers testés :")
print(f"  - Codables (au moins une solution trouvée) : {succes_count}")
print(f"  - Inodables (bloqués) : {echec_count}")
print(f"  - Taux de réussite : {(succes_count / len(primes)) * 100:.2f}%")