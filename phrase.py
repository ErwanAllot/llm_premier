import math

BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

blocks = []
for val, b_str, barre_str in BASE_TOKENS:
    blocks.append(b_str)
    blocks.append(barre_str)

unique_blks = sorted(list(set(blocks)), key=len, reverse=True)

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0: return False
    return True

def get_all_partitions(sub_s):
    """Renvoie TOUTES les partitions possibles d'une chaîne binaire."""
    if not sub_s:
        return [[]]
    results = []
    for b in unique_blks:
        if b and sub_s.startswith(b):
            for p in get_all_partitions(sub_s[len(b):]):
                results.append([b] + p)
    return results

# Générer quelques nombres premiers
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

print("Recherche des liaisons 'premières' entre nombres premiers consécutifs...")

# Pour chaque paire de nombres premiers consécutifs
for i in range(len(primes) - 1):
    p1, p2 = primes[i], primes[i+1]
    s1, s2 = bin(p1)[2:], bin(p2)[2:]
    
    parts1 = get_all_partitions(s1)
    parts2 = get_all_partitions(s2)
    
    if not parts1 or not parts2:
        continue
        
    # Tester les combinaisons de fins et de débuts
    bridging_tokens = set()
    for pt1 in parts1:
        t_end = pt1[-1] # Dernier token du premier nombre
        for pt2 in parts2:
            t_start = pt2[0] # Premier token du second nombre
            
            # Concaténation binaire des deux tokens
            bridge_bin = t_end + t_start
            bridge_val = int(bridge_bin, 2)
            
            if is_prime(bridge_val):
                bridging_tokens.add((t_end, t_start, bridge_val))
                
    if bridging_tokens:
        print(f"Prime {p1} -> {p2} : {len(bridging_tokens)} pont(s) trouvé(s)")
        for t_e, t_s, val in bridging_tokens:
            print(f"   Fin '{t_e}' + Début '{t_s}' = binaire {t_e+t_s} (décimal {val}, qui est premier !)")