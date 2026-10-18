def get_barre(b_str):
    """Génère le bitwise barré d'une chaîne binaire."""
    return "".join('1' if c == '0' else '0' for c in b_str)

# Les rebelles et leurs représentations binaires
rebelles_dict = {
    4: "100",
    6: "110",
    25: "11001",
    27: "11011",
    100: "1100100",
    102: "1100110",
    411: "110011011"
}

rebelles_tokens = list(rebelles_dict.values())
rebelles_tokens_barre = [get_barre(b) for b in rebelles_tokens]
all_rebel_blocks = rebelles_tokens + rebelles_tokens_barre

def can_partition(sub_s, blks, memo=None):
    if memo is None:
        memo = {}
    if not sub_s:
        return True
    if sub_s in memo:
        return memo[sub_s]
    
    unique_blks = sorted(list(set(blks)), key=len, reverse=True)
    for b in unique_blks:
        if b and sub_s.startswith(b):
            if can_partition(sub_s[len(b):], blks, memo):
                memo[sub_s] = True
                return True
                
    memo[sub_s] = False
    return False

# Testons ce que ce nouvel alphabet arrive à coder
max_test = 1000
echoues = []

for n in range(1, max_test + 1):
    s = bin(n)[2:]
    if not can_partition(s, all_rebel_blocks):
        echoues.append((n, s))

print(f"=== TEST DU NOUVEL ALPHABET (Rebelles + Barrés) sur {max_test} entiers ===")
if echoues:
    print(f"Premiers nombres impossibles à coder ({len(echoues)} échecs au total) :")
    for n, s in echoues[:max_test]:
        print(f"  - {n} (binaire : {s})")
else:
    print("Tous les nombres sont codables !")