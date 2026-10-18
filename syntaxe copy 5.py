BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

# On extrait toutes les briques valides du dictionnaire
VALID_BRICKS = set()
for val, b1, b2 in BASE_TOKENS:
    VALID_BRICKS.add(b1)
    VALID_BRICKS.add(b2)

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def decompose_number(n_bin):
    """Découpe une chaîne binaire en briques valides (Glouton + Backtracking)."""
    if n_bin in VALID_BRICKS:
        return [n_bin]
    
    # On teste les préfixes du plus grand au plus petit
    max_len = max(len(b) for b in VALID_BRICKS)
    for i in range(min(len(n_bin), max_len), 0, -1):
        prefix = n_bin[:i]
        if prefix in VALID_BRICKS:
            suffix = n_bin[i:]
            if not suffix:
                return [prefix]
            sub_res = decompose_number(suffix)
            if sub_res is not None:
                return [prefix] + sub_res
    return None

def format_prime(p):
    n_bin = bin(p)[2:]
    bricks = decompose_number(n_bin)
    
    if bricks is None:
        brick_str = f"({n_bin})" # Sécurité si non couverte
    else:
        brick_str = "".join([f"({b})" for b in bricks])
        
    if p == 2:
        return f"<[&,&]{brick_str}>"
    if p == 3:
        return f"<[&,%]{brick_str}>"
    
    m6 = "+" if p % 6 == 1 else "-"
    m4 = "*" if p % 4 == 1 else "%"
    
    return f"<[{m6},{m4}]{brick_str}>"

def generate_corpus(max_limit):
    all_primes = [n for n in range(2, max_limit + 1) if is_prime(n)]
    corpus_elements = []
    
    i = 0
    while i < len(all_primes):
        p = all_primes[i]
        
        # Détection des triplets (ex: 3, 5, 7)
        if i + 2 < len(all_primes) and all_primes[i+1] == p + 2 and all_primes[i+2] == p + 4:
            group = "".join([format_prime(all_primes[i+j]) for j in range(3)])
            corpus_elements.append(f"{{{group}}}")
            i += 3
        # Détection des jumeaux (p, p+2)
        elif i + 1 < len(all_primes) and all_primes[i+1] == p + 2:
            group = "".join([format_prime(all_primes[i+j]) for j in range(2)])
            corpus_elements.append(f"{{{group}}}")
            i += 2
        else:
            corpus_elements.append(format_prime(p))
            i += 1
            
    return "".join(corpus_elements)

# Test de génération propre jusqu'à 20
print("--- CORPUS PROPRE (SANS BALISES LETTRES) ---")
print(generate_corpus(20))