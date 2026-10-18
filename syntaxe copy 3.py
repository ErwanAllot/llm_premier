def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def get_bricks(n):
    b = bin(n)[2:]
    chunks = [b[i:i+2] for i in range(0, len(b), 2)]
    return "".join([f"({c})" for c in chunks])

def format_prime(p, letter_open="", letter_close=""):
    if p == 2:
        return f"<[&,&]{letter_open}{get_bricks(p)}{letter_close}>"
    if p == 3:
        return f"<[&,%]{letter_open}{get_bricks(p)}{letter_close}>"
    
    m6 = "+" if p % 6 == 1 else "-"
    m4 = "*" if p % 4 == 1 else "%"
    
    return f"<[{m6},{m4}]{letter_open}{get_bricks(p)}{letter_close}>"

# Génération des nombres premiers jusqu'à 31
max_limit = 31
all_primes = [n for n in range(2, max_limit + 1) if is_prime(n)]

corpus_elements = []
i = 0
while i < len(all_primes):
    p = all_primes[i]
    
    # Détection des triplets de type (3, 5, 7) ou regroupements spéciaux
    if i + 2 < len(all_primes) and all_primes[i+1] == p + 2 and all_primes[i+2] == p + 4:
        # Triplet (ex: 3, 5, 7)
        p1 = format_prime(all_primes[i], "A", "a")
        p2 = format_prime(all_primes[i+1], "A", "a")
        p3 = format_prime(all_primes[i+2], "A", "a")
        corpus_elements.append(f"{{{p1}{p2}{p3}}}")
        i += 3
    elif i + 1 < len(all_primes) and all_primes[i+1] == p + 2:
        # Jumeaux classiques (p, p+2)
        p1 = format_prime(all_primes[i], "B", "b")
        p2 = format_prime(all_primes[i+1], "B", "b")
        corpus_elements.append(f"{{{p1}{p2}}}")
        i += 2
    else:
        # Nombre isolé avec une balise de portée si nécessaire
        corpus_elements.append(format_prime(p))
        i += 1

print("--- FLUX CORPUS CORRIGÉ ---")
print("".join(corpus_elements))