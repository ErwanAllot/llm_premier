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

def generate_inter_prime_bridges():
    primes = []
    c = 2
    while len(primes) < 30:
        if is_prime(c):
            primes.append(c)
        c += 1

    print("--- PONTS GIGOGNES INTER-PREMIERS ---")
    print("-" * 55)
    
    # Dictionnaire pour se souvenir de quel premier provient quelle brique
    brick_history = {} # brique -> prime d'origine

    for p in primes:
        mod6 = p % 6
        mod4 = p % 4
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        
        if bricks is None:
            bricks_str = "(néant)"
        else:
            n_tokens = len(bricks)
            
            # Vérifions si les briques de ce nombre proviennent d'un assemblage de briques antérieures
            is_built_from_past = all(b in brick_history for b in bricks)
            
            if n_tokens >= 2 and is_built_from_past:
                # Si les briques viennent du passé, on active la balise gigogne pour souligner le pont
                open_tag = chr(64 + min(n_tokens - 1, 26))
                close_tag = chr(96 + min(n_tokens - 1, 26))
                inner = "".join([f"({b})" for b in bricks])
                bricks_str = f"{open_tag}{inner}{close_tag}"
            else:
                bricks_str = "".join([f"({b})" for b in bricks])
                
            # Enregistrement des briques de ce nombre premier pour le futur
            for b in bricks:
                if b not in brick_history:
                    brick_history[b] = p
                
        syntax_line = f"<[{mod6}, {mod4}]{bricks_str}>"
        print(f"p = {p:<3} -> {syntax_line}")

if __name__ == "__main__":
    generate_inter_prime_bridges()