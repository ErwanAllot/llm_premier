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

def generate_full_gigogne_syntax():
    primes = []
    c = 2
    while len(primes) < 30:
        if is_prime(c):
            primes.append(c)
        c += 1

    print("--- SYNTAXE GIGOGNE GÉNÉRALISÉE (TOUS LES TOKENS) ---")
    print("-" * 55)
    
    for p in primes:
        mod6 = p % 6
        mod4 = p % 4
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        
        if bricks is None:
            bricks_str = "(néant)"
        else:
            n_tokens = len(bricks)
            # Si le nombre est composé de 2 tokens ou plus, on encadre par la lettre correspondante (A/a, B/b, C/c...)
            if n_tokens >= 2:
                open_tag = chr(64 + n_tokens - 1)  # 2->A, 3->B, 4->C, etc.
                close_tag = chr(96 + n_tokens - 1) # 2->a, 3->b, 4->c, etc.
                inner = "".join([f"({b})" for b in bricks])
                bricks_str = f"{open_tag}{inner}{close_tag}"
            else:
                bricks_str = f"({bricks[0]})"
                
        syntax_line = f"<[{mod6}, {mod4}]{bricks_str}>"
        print(f"p = {p:<3} -> {syntax_line}")

if __name__ == "__main__":
    generate_full_gigogne_syntax()