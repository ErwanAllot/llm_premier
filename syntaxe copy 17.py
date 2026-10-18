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

def get_tag_by_length(n):
    mapping = {
        2: ('A', 'a'),
        3: ('B', 'b'),
        4: ('C', 'c'),
        5: ('D', 'd'),
        6: ('E', 'e')
    }
    return mapping.get(n, (None, None))

def generate_strict_ribbon_syntax():
    primes = []
    c = 2
    while len(primes) < 30:
        if is_prime(c):
            primes.append(c)
        c += 1

    prime_data = []
    global_tokens = []

    for p in primes:
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        prime_data.append({'p': p, 'mod6': p % 6, 'mod4': p % 4, 'bricks': bricks})
        if bricks:
            for b in bricks:
                global_tokens.append({'token': b, 'prime_val': p})

    open_tags = [[] for _ in range(len(global_tokens))]
    close_tags = [[] for _ in range(len(global_tokens))]

    # Balayage du ruban avec des fenêtres de 2 à 6 tokens
    for N in range(2, 7):
        for i in range(len(global_tokens) - N + 1):
            window = global_tokens[i:i+N]
            concat_bin = "".join([t['token'] for t in window])
            try:
                val = int(concat_bin, 2)
                if is_prime(val):
                    open_char, close_char = get_tag_by_length(N)
                    if open_char and open_char not in open_tags[i]:
                        open_tags[i].append(open_char)
                    if close_char and close_char not in close_tags[i + N - 1]:
                        close_tags[i + N - 1].append(close_char)
            except ValueError:
                pass

    print("--- RUBAN CONTINU (BALISES STRICTES PAR TAILLE) ---")
    print("-" * 55)
    
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
    generate_strict_ribbon_syntax()