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

unique_blks = sorted(list(set(blocks)), key=len, reverse=True)

def find_partition(sub_s):
    if not sub_s:
        return []
    for b in unique_blks:
        if b and sub_s.startswith(b):
            rest = find_partition(sub_s[len(b):])
            if rest is not None:
                return [b] + rest
    return None

s_test = bin(923)[2:] # "1110011011"
partition = find_partition(s_test)
print(f"Partition de {s_test} (923) : {partition}")