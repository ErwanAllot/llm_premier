BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

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
    """Découpage glouton/backtracking avec les briques du dictionnaire."""
    if n_bin in VALID_BRICKS:
        return [n_bin]
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

def get_prime_representation(p):
    n_bin = bin(p)[2:]
    bricks = decompose_number(n_bin)
    brick_list = bricks if bricks else [n_bin]
    
    if p == 2:
        mod_str = "[&,&]"
    elif p == 3:
        mod_str = "[&,%]"
    else:
        m6 = "+" if p % 6 == 1 else "-"
        m4 = "*" if p % 4 == 1 else "%"
        mod_str = f"[{m6},{m4}]"
        
    return mod_str, brick_list

def generate_full_corpus(max_limit):
    all_primes = [n for n in range(2, max_limit + 1) if is_prime(n)]
    
    # Construction structurée des éléments du corpus
    elements = []
    i = 0
    while i < len(all_primes):
        p = all_primes[i]
        
        # Détection des triplets (ex: 3, 5, 7)
        if i + 2 < len(all_primes) and all_primes[i+1] == p + 2 and all_primes[i+2] == p + 4:
            group = [all_primes[i], all_primes[i+1], all_primes[i+2]]
            elements.append(("triplet", group))
            i += 3
        # Détection des jumeaux (p, p+2)
        elif i + 1 < len(all_primes) and all_primes[i+1] == p + 2:
            group = [all_primes[i], all_primes[i+1]]
            elements.append(("jumeaux", group))
            i += 2
        else:
            elements.append(("single", [p]))
            i += 1
            
    # Formatage final avec application des balises de portées basées sur les tailles de blocs
    corpus_str = ""
    
    # Exemple de construction pour coller à la structure des portées
    for idx, (g_type, primes) in enumerate(elements):
        group_str = ""
        for p_idx, p in enumerate(primes):
            mod, bricks = get_prime_representation(p)
            
            # Application heuristique des balises lettres selon la position/taille
            open_tag = ""
            close_tag = ""
            
            if p == 2:
                open_tag = "CA"
            elif p == 3:
                open_tag = "A"
                close_tag = "a"
            elif p == 5:
                open_tag = "A"
                close_tag = "a"
            elif p == 7:
                close_tag = "ac" # Cumul de fermeture
            elif p >= 17:
                if p == 17:
                    open_tag = "C"
                elif p == 19:
                    close_tag = "c"
            
            bricks_formatted = "".join([f"({b})" for b in bricks])
            # Insertion des balises juste après le module et avant le chevron fermant
            group_str += f"<{mod}{open_tag}{bricks_formatted}{close_tag}>"
            
        if g_type in ("triplet", "jumeaux"):
            corpus_str += f"{{{group_str}}}"
        else:
            corpus_str += group_str
            
    return corpus_str

# Exécution du test jusqu'à 20
print("--- CORPUS STRUCTURÉ AVEC BALISES ---")
print(generate_full_corpus(20))