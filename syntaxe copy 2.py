def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def get_bricks(n):
    # Conversion en binaire et découpage en morceaux (briques) avec des parenthèses ()
    b = bin(n)[2:]
    chunks = [b[i:i+2] for i in range(0, len(b), 2)]
    return "".join([f"({c})" for c in chunks])

def format_prime(p):
    if p == 2 or p == 3:
        return f"<[&,&]{get_bricks(p)}>"
    
    # Modulo 6 (+ pour 1 mod 6, - pour 5 mod 6)
    m6 = "+" if p % 6 == 1 else "-"
    # Modulo 4 (* pour 1 mod 4, % pour 3 mod 4)
    m4 = "*" if p % 4 == 1 else "%"
    
    return f"<[{m6},{m4}]{get_bricks(p)}>"

# Génération des nombres premiers jusqu'à 45 pour l'exemple
max_limit = 45
all_primes = [n for n in range(2, max_limit + 1) if is_prime(n)]

corpus_elements = []
i = 0
while i < len(all_primes):
    p = all_primes[i]
    # Vérification des nombres premiers jumeaux (p et p+2)
    if i + 1 < len(all_primes) and all_primes[i+1] == p + 2:
        p1_str = format_prime(p)
        p2_str = format_prime(all_primes[i+1])
        # Les jumeaux sont enveloppés dans {}
        corpus_elements.append(f"{{{p1_str}{p2_str}}}")
        i += 2
    else:
        corpus_elements.append(format_prime(p))
        i += 1

# Résultat final combiné en un flux de texte continu
final_corpus = "".join(corpus_elements)
print("--- FLUX CORPUS GÉNÉRÉ ---")
print(final_corpus)