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

# Mappage des tailles de fenêtres aux lettres de balises
# 2 tokens -> Aa (lettre index 0 -> 'A', 'a')
# 3 tokens -> Bb (lettre index 1 -> 'B', 'b')
# 4 tokens -> Cc (lettre index 2 -> 'C', 'c')
SPAN_LETTERS = {
    2: ('A', 'a'),
    3: ('B', 'b'),
    4: ('C', 'c'),
    5: ('D', 'd')
}

def generate_corpus_algorithmic(max_limit):
    all_primes = [n for n in range(2, max_limit + 1) if is_prime(n)]
    
    # 1. Construction de la liste plate de tous les blocs de nombres premiers et leurs briques
    prime_structures = []
    flat_tokens = [] # Liste de dictionnaires pour suivre chaque brique individuelle
    
    i = 0
    while i < len(all_primes):
        p = all_primes[i]
        if i + 2 < len(all_primes) and all_primes[i+1] == p + 2 and all_primes[i+2] == p + 4:
            group = [all_primes[i], all_primes[i+1], all_primes[i+2]]
            g_type = "triplet"
            i += 3
        elif i + 1 < len(all_primes) and all_primes[i+1] == p + 2:
            group = [all_primes[i], all_primes[i+1]]
            g_type = "jumeaux"
            i += 2
        else:
            group = [p]
            g_type = "single"
            
        group_items = []
        for gp in group:
            n_bin = bin(gp)[2:]
            bricks = decompose_number(n_bin) or [n_bin]
            
            if gp == 2: mod_str = "[&,&]"
            elif gp == 3: mod_str = "[&,%]"
            else:
                m6 = "+" if gp % 6 == 1 else "-"
                m4 = "*" if gp % 4 == 1 else "%"
                mod_str = f"[{m6},{m4}]"
                
            item = {
                "prime": gp,
                "mod": mod_str,
                "bricks": bricks,
                "open_tags": [],
                "close_tags": []
            }
            group_items.append(item)
            # On enregistre chaque brique dans le flux plat pour l'analyse de voisinage
            for b in bricks:
                flat_tokens.append({"brick": b, "parent_item": item})
                
        prime_structures.append((g_type, group_items))

    # 2. Analyse algorithmique des fenêtres glissantes (voisinage) pour placer les balises
    # Pour chaque taille de portée (ex: 2, 3, 4 tokens)
    for span_size, (maj, min_lettre) in SPAN_LETTERS.items():
        for start_idx in range(len(flat_tokens) - span_size + 1):
            window = flat_tokens[start_idx : start_idx + span_size]
            # Concaténation des briques de la fenêtre
            concat_bin = "".join([t["brick"] for t in window])
            concat_val = int(concat_bin, 2)
            
            # Si le nombre combiné est premier, on valide cette portée !
            if is_prime(concat_val):
                # La majuscule s'ouvre sur le premier token de la fenêtre
                window[0]["parent_item"]["open_tags"].append(maj)
                # La minuscule se ferme sur le dernier token de la fenêtre
                window[-1]["parent_item"]["close_tags"].append(min_lettre)

    # 3. Assemblage final de la chaîne du corpus
    corpus_str = ""
    for g_type, group_items in prime_structures:
        group_str = ""
        for item in group_items:
            # Concaténation des balises d'ouverture et de fermeture
            open_str = "".join(item["open_tags"])
            close_str = "".join(item["close_tags"])
            bricks_str = "".join([f"({b})" for b in item["bricks"]])
            
            group_str += f"<{item['mod']}{open_str}{bricks_str}{close_str}>"
            
        if g_type in ("triplet", "jumeaux"):
            corpus_str += f"{{{group_str}}}"
        else:
            corpus_str += group_str
            
    return corpus_str

print("--- CORPUS AVEC ALGORITHME DE BALISAGE DYNAMIQUE ---")
print(generate_corpus_algorithmic(20))