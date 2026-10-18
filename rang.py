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

def generate_first_n_primes(n_count):
    primes = []
    candidate = 2
    while len(primes) < n_count:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    return primes

def print_prime_tokens_report():
    primes = generate_first_n_primes(100)
    
    print(f"{'Rang':<6} | {'Token(Rang)':<25} | {'Premier':<8} | {'Token(Premier)'}")
    print("-" * 75)
    
    for idx, p in enumerate(primes, start=1):
        # Décomposition de la valeur du rang
        idx_bin = bin(idx)[2:]
        idx_bricks = decompose_number(idx_bin)
        idx_token_str = " + ".join([f"({b})" for b in idx_bricks]) if idx_bricks else "néant"
        
        # Décomposition du nombre premier
        p_bin = bin(p)[2:]
        p_bricks = decompose_number(p_bin)
        p_token_str = " + ".join([f"({b})" for b in p_bricks]) if p_bricks else "néant"
            
        print(f"{idx:<6} | {idx_token_str:<25} | {p:<8} | {p_token_str}")

if __name__ == "__main__":
    print_prime_tokens_report()