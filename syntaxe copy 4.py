def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def get_bricks(n):
    b = bin(n)[2:]
    chunks = [b[i:i+2] for i in range(0, len(b), 2)]
    return "".join([f"({c})" for c in chunks])

def format_prime(p):
    if p == 2:
        return f"<[&,&]{get_bricks(p)}>"
    if p == 3:
        return f"<[&,%]{get_bricks(p)}>"
    
    m6 = "+" if p % 6 == 1 else "-"
    m4 = "*" if p % 4 == 1 else "%"
    
    return f"<[{m6},{m4}]{get_bricks(p)}>"

def generate_clean_corpus(max_limit):
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

# Test propre jusqu'à 20 pour voir
print("--- CORPUS PROPRE SANS BALISES ---")
print(generate_clean_corpus(20))