BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

# Séparation des familles
tokens_only = [b_str for _, b_str, _ in BASE_TOKENS]
barres_only = [barre_str for _, _, barre_str in BASE_TOKENS]
all_blocks = tokens_only + barres_only

rebelles = {
    4: "100",
    6: "110",
    25: "11001",
    27: "11011",
    100: "1100100",
    102: "1100110",
    411: "110011011"
}

def can_partition_with(sub_s, blks, memo=None):
    if memo is None:
        memo = {}
    if not sub_s:
        return True
    if sub_s in memo:
        return memo[sub_s]
    
    unique_blks = sorted(list(set(blks)), key=len, reverse=True)
    for b in unique_blks:
        if b and sub_s.startswith(b):
            if can_partition_with(sub_s[len(b):], blks, memo):
                memo[sub_s] = True
                return True
                
    memo[sub_s] = False
    return False

print("--- ANALYSE DES REBELLES ---")
for n, s in rebelles.items():
    ok_tokens = can_partition_with(s, tokens_only)
    ok_barres = can_partition_with(s, barres_only)
    ok_both = can_partition_with(s, all_blocks)
    
    print(f"Nombre {n} ({s}) :")
    print(f"  -> Tokens seuls : {'OUI' if ok_tokens else 'NON'}")
    print(f"  -> Barrés seuls : {'OUI' if ok_barres else 'NON'}")
    print(f"  -> Mixte (les deux) : {'OUI' if ok_both else 'NON'}")