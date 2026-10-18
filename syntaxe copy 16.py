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

def generate_ribbon_syntax():
    # 1. Génération des premiers et de leur décomposition en tokens
    primes = []
    c = 2
    while len(primes) < 30:
        if is_prime(c):
            primes.append(c)
        c += 1

    prime_data = []
    global_tokens = [] # Liste plate de tous les tokens du ruban

    for p in primes:
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        prime_data.append({'p': p, 'mod6': p % 6, 'mod4': p % 4, 'bricks': bricks})
        if bricks:
            for b in bricks:
                global_tokens.append({'token': b, 'prime_val': p})

    # 2. Initialisation des listes de balises pour chaque position du ruban global
    open_tags = [[] for _ in range(len(global_tokens))]
    close_tags = [[] for _ in range(len(global_tokens))]

    tag_counter = 0
    tag_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # 3. Balayage du ruban continu pour toutes les tailles de fenêtres (N >= 2)
    for N in range(2, 5):  # Fenêtres de 2, 3 et 4 tokens
        for i in range(len(global_tokens) - N + 1):
            window = global_tokens[i:i+N]
            concat_bin = "".join([t['token'] for t in window])
            try:
                val = int(concat_bin, 2)
                if is_prime(val):
                    letter = tag_letters[tag_counter % len(tag_letters)]
                    tag_counter += 1
                    open_tags[i].append(letter)
                    close_tags[i + N - 1].append(letter.lower())
            except ValueError:
                pass

    # 4. Reconstruction et affichage formaté par nombre premier
    print("--- RUBAN CONTINU DE TOKENS ET BALISES TRANSVERSALES ---")
    print("-" * 60)
    
    token_global_idx = 0
    for item in prime_data:
        p = item['p']
        mod6 = item['mod6']
        mod4 = item['mod4']
        bricks = item['bricks']
        
        if bricks is None:
            bricks_str = "(néant)"
        else:
            brick_strs = []
            for b in bricks:
                o_str = "".join(open_tags[token_global_idx])
                c_str = "".join(close_tags[token_global_idx])
                token_global_idx += 1
                brick_strs.append(f"{o_str}({b}){c_str}")
            bricks_str = "".join(brick_strs)
            
        print(f"p = {p:<3} -> <[{mod6}, {mod4}]{bricks_str}>")

if __name__ == "__main__":
    generate_ribbon_syntax()