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

def generate_gigogne_syntax():
    primes = []
    c = 2
    while len(primes) < 25:
        if is_prime(c):
            primes.append(c)
        c += 1

    # Pré-calcul des structures
    parsed = []
    for p in primes:
        mod6 = p % 6
        mod4 = p % 4
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        parsed.append({'p': p, 'mod6': mod6, 'mod4': mod4, 'bricks': bricks})

    # Attribution déterministe des balises gigognes sur les briques
    # On parcourt les briques pour repérer les ouvertures/fermetures de motifs
    tags_open = [[] for _ in range(len(parsed))]
    tags_close = [[] for _ in range(len(parsed))]
    
    tag_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    tag_idx = 0

    # Recherche de paires de briques identiques à distance (fenêtre max 6)
    for i in range(len(parsed)):
        if not parsed[i]['bricks']: continue
        for j in range(i + 1, min(i + 7, len(parsed))):
            if not parsed[j]['bricks']: continue
            # Si le premier token de la brique i correspond au premier token de j
            if parsed[i]['bricks'][0] == parsed[j]['bricks'][0]:
                tag = tag_chars[tag_idx % len(tag_chars)]
                tag_idx += 1
                tags_open[i].append(tag)
                tags_close[j].append(tag.lower())
                break # Une balise par correspondance locale

    print("--- SYNTAXE STRUCTURÉE AVEC BALISES GIGOGNES ---")
    print("-" * 55)
    
    for idx, item in enumerate(parsed):
        p = item['p']
        mod6 = item['mod6']
        mod4 = item['mod4']
        bricks = item['bricks']
        
        open_str = "".join(tags_open[idx])
        close_str = "".join(tags_close[idx])
        
        if bricks is None:
            bricks_str = "(néant)"
        else:
            bricks_str = "".join([f"({b})" for b in bricks])
            
        # Intégration des balises autour du bloc de briques
        syntax_line = f"<[{mod6}, {mod4}]{open_str}{bricks_str}{close_str}>"
        print(f"p = {p:<3} -> {syntax_line}")

if __name__ == "__main__":
    generate_gigogne_syntax()