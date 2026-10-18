from sympy import isprime

# Vos tokens irréductibles de base actuels (en décimal)
tokens_val = [2, 3, 5, 7, 17, 19, 409]

def get_barre(s):
    return "".join('1' if c == '0' else '0' for c in s)

# Convertir en binaire
tokens_bin = [bin(v)[2:] for v in tokens_val]

def can_partition(sub_s, blks):
    if not sub_s: return True
    unique_blks = sorted(list(set(blks)), key=len, reverse=True)
    for b in unique_blks:
        if b and sub_s.startswith(b):
            if can_partition(sub_s[len(b):], blks):
                return True
    return False

print(f"Tokens actuels : {tokens_val}")
print("Recherche des prochains tokens irréductibles au-delà de 409...\n")

# On cherche les prochains nombres premiers après 409
n = 410
nouveaux_tokens_trouves = 0
limite_recherche = 20000000  # On cherche jusqu'à un certain point

while nouveaux_tokens_trouves < 5 and n < limite_recherche:
    if isprime(n):
        s = bin(n)[2:]
        blocks = []
        for t in tokens_bin:
            blocks.append(t)
            blocks.append(get_barre(t))
        
        # Si le nombre premier NE PEUT PAS être formé par nos briques actuelles...
        if not can_partition(s, blocks):
            tokens_bin.append(s)
            tokens_val.append(n)
            nouveaux_tokens_trouves += 1
            print(f"🎉 Nouveau token irréductible trouvé : {n} (binaire: {s})")
            print(f"   Liste mise à jour : {tokens_val}\n")
    n += 1

print("Recherche terminée.")