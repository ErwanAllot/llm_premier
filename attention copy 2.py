BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

# Les deux familles distinctes
rebelles_tokens = [b_str for _, b_str, _ in BASE_TOKENS]
rebelles_tokens_barre = [barre_str for _, _, barre_str in BASE_TOKENS]

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

# Testons les nombres naturels de 1 jusqu'à 200
max_test = 200
echoues_tokens = []
echoues_barres = []

for n in range(1, max_test + 1):
    s = bin(n)[2:]
    if not can_partition(s, rebelles_tokens):
        echoues_tokens.append((n, s))
    if not can_partition(s, rebelles_tokens_barre):
        echoues_barres.append((n, s))

print(f"=== TEST SUR LES {max_test} PREMIERS ENTIERS NATURELS ===")
print(f"Nombres non constructibles avec UNIQUEMENT rebelles_tokens :")
print(echoues_tokens[:20] if echoues_tokens else "Aucun ! (tout passe)")

print(f"\nNombres non constructibles avec UNIQUEMENT rebelles_tokens_barre :")
print(echoues_barres[:20] if echoues_barres else "Aucun ! (tout passe)")