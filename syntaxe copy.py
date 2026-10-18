def is_prime(num):
    if num < 2: return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0: return False
    return True

# Briques de base simplifiées pour l'exemple
BASE_TOKENS = [
    (2, "10", "01"), (3, "11", "00"), (5, "101", "010"),
    (7, "111", "000"), (11, "1011", "0100"), (13, "1101", "0010")
]

# Fonction de décomposition simplifiée (gloutonne)
def get_bricks(n_bin):
    # Pour l'exemple, on découpe grossièrement par morceaux de 2 ou 3
    chunks = [n_bin[i:i+3] for i in range(0, len(n_bin), 3)]
    return chunks

# Test sur les premiers nombres jusqu'à 23
primes = [p for p in range(5, 30) if is_prime(p)]

print("--- VISUALISATION DE LA SYNTAXE ---")
for i, p in enumerate(primes):
    p_bin = bin(p)[2:]
    bricks = get_bricks(p_bin)
    
    # Modulo 6 et Modulo 4
    m6 = "+" if p % 6 == 1 else "-"
    m4 = "*" if p % 4 == 1 else "%"
    
    # Jumeaux (si p-2 ou p+2 est premier)
    is_twin = is_prime(p - 2) or is_prime(p + 2)
    
    # Construction de la chaîne selon vos règles
    # Format : { < [mod6, mod4] A (brick),(brick) a > }
    mod_str = f"[{m6},{m4}]"
    inner_bricks = ",".join([f"({b})" for b in bricks])
    
    # Ajout d'une balise lettre A...a juste après les modulos si la structure s'y prête
    letter_open = "A" if i % 2 == 0 else "B"
    letter_close = letter_open.lower()
    
    body = f"<{mod_str}{letter_open}{inner_bricks}{letter_close}>"
    
    if is_twin:
        formatted_prime = f"{{{body}}}"
    else:
        formatted_prime = body
        
    print(f"Prime {p:2d} -> {formatted_prime}")


    # <[&,&](10)>{<[&,%](11)><[-,*](101)><[+,%](111)>}{<[-,%](101)(1)><[+,*](110)(1)>}
    # <[&,&]CA(10)>{<[&,%]A(11)a><[-,*]A(101)a><[+,%](111)ac>}{<[-,%](101)(1)><[+,*](110)(1)>}{<[-,*]C(100)(01)><[+,%](100)(11)c>}