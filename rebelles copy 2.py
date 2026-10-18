# les rebelles
# 4 : 100
# 6 : 110
# 25 : 11001
# 27 : 11011
# 100 : 1100100
# 102 : 1100110
# 411 : 110011011


# Définition de vos 7 tokens de base avec leurs binaires et leurs barrés
BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

blocks = []
for val, b_str, barre_str in BASE_TOKENS:
    blocks.append(b_str)
    blocks.append(barre_str)

# On enlève les doublons une bonne fois pour toutes
unique_blks = sorted(list(set(blocks)), key=len, reverse=True)

def can_partition(sub_s, memo=None):
    if memo is None:
        memo = {}
    if not sub_s:
        return True
    if sub_s in memo:
        return memo[sub_s]
    
    for b in unique_blks:
        if b and sub_s.startswith(b):
            if can_partition(sub_s[len(b):], memo):
                memo[sub_s] = True
                return True
                
    memo[sub_s] = False
    return False

# Testons notre fameux 923 (1110011011)
s_test = bin(923)[2:]
print(f"Test de 923 ({s_test}) : partitionnable ? {can_partition(s_test)}")