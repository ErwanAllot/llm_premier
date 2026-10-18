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

# Mapping de la distance en tokens vers les paires de balises (Maj, min)
# 2 tokens de distance -> Aa, 3 -> Bb, 4 -> Cc, 5 -> Dd, etc.
def get_letter_for_distance(distance):
    if distance < 2:
        return None
    # ASCII 'A' = 65. Distance 2 -> 'A' (index 0)
    char_code = ord('A') + (distance - 2)
    if char_code > ord('Z'):
        return None # Sécurité si on dépasse l'alphabet
    maj = chr(char_code)
    min_lettre = maj.lower()
    return (maj, min_lettre)

def build_prime_syntax_corpus(max_limit):
    all_primes = [n for n in range(2, max_limit + 1) if is_prime(n)]
    
    # 1. CONSTRUCTION DE L'OBJET STRUCTURE / GROUPES
    # On segmente en single, jumeaux ou triplets
    groups = []
    flat_tokens_stream = [] # Références vers tous les tokens dans l'ordre séquentiel global
    
    i = 0
    while i < len(all_primes):
        p = all_primes[i]
        if i + 2 < len(all_primes) and all_primes[i+1] == p + 2 and all_primes[i+2] == p + 4:
            prime_list = [all_primes[i], all_primes[i+1], all_primes[i+2]]
            g_type = "triplet"
            i += 3
        elif i + 1 < len(all_primes) and all_primes[i+1] == p + 2:
            prime_list = [all_primes[i], all_primes[i+1]]
            g_type = "jumeaux"
            i += 2
        else:
            prime_list = [p]
            g_type = "single"
            
        group_obj = {"type": g_type, "primes": []}
        
        for gp in prime_list:
            n_bin = bin(gp)[2:]
            bricks = decompose_number(n_bin) or [n_bin]
            
            if gp == 2: mod_str = "[&,&]"
            elif gp == 3: mod_str = "[&,%]"
            else:
                m6 = "+" if gp % 6 == 1 else "-"
                m4 = "*" if gp % 4 == 1 else "%"
                mod_str = f"[{m6},{m4}]"
                
            # Structure de l'objet pour ce nombre premier
            prime_node = {
                "prime": gp,
                "mod": mod_str,
                "token_count": len(bricks),
                "tokens": [] # Tableau contenant chaque brique avec ses sous-tableaux [ouvertures, fermetures]
            }
            
            for b in bricks:
                token_entry = {
                    "brick": b,
                    "openings": [],   # Tableau d'ouvertures
                    "closings": []    # Tableau de fermetures
                }
                prime_node["tokens"].append(token_entry)
                # On pousse aussi la référence dans le flux plat global pour calculer les distances
                flat_tokens_stream.append(token_entry)
                
            group_obj["primes"].append(prime_node)
        groups.append(group_obj)

    # 2. ANALYSE DE DISTANCE ET PLACEMENT DES BALISES
    # On parcourt toutes les combinaisons de fenêtres [start_idx ... end_idx] dans le flux plat
    total_tokens = len(flat_tokens_stream)
    for start_idx in range(total_tokens):
        for end_idx in range(start_idx, total_tokens):
            distance = (end_idx - start_idx) + 1
            letter_pair = get_letter_for_distance(distance)
            
            if letter_pair:
                maj, min_lettre = letter_pair
                # Concaténation des briques de la fenêtre
                window_bricks = [flat_tokens_stream[k]["brick"] for k in range(start_idx, end_idx + 1)]
                concat_bin = "".join(window_bricks)
                concat_val = int(concat_bin, 2)
                
                # Si le nombre réuni est premier, on place les lettres !
                if is_prime(concat_val):
                    flat_tokens_stream[start_idx]["openings"].append(maj)
                    flat_tokens_stream[end_idx]["closings"].append(min_lettre)

    # 3. TRI DES TABLEAUX D'OUVERTURE (Z à A) ET FERMETURE (a à z)
    for token_entry in flat_tokens_stream:
        # Ouvertures de Z vers A (pour structurer la hiérarchie imbriquée)
        token_entry["openings"].sort(reverse=True)
        # Fermetures de a vers z
        token_entry["closings"].sort()

    # 4. RECOMPOSITION FINALE DE LA PARTITION
    corpus_str = ""
    for group_obj in groups:
        group_str = ""
        for prime_node in group_obj["primes"]:
            # Construction des briques de ce nombre premier avec leurs balises respectives
            bricks_formatted = ""
            for token_entry in prime_node["tokens"]:
                op_str = "".join(token_entry["openings"])
                cl_str = "".join(token_entry["closings"])
                bricks_formatted += f"{op_str}({token_entry['brick']}){cl_str}"
                
            group_str += f"<{prime_node['mod']}{bricks_formatted}>"
            
        if group_obj["type"] in ("triplet", "jumeaux"):
            corpus_str += f"{{{group_str}}}"
        else:
            corpus_str += group_str
            
    return corpus_str

# Test d'exécution
print("--- CORPUS STRUCTURÉ ARCHITECTURE OBJET & DISTANCE ---")
print(build_prime_syntax_corpus(20))