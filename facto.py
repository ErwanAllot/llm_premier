from sympy import factorint

# Copions quelques-uns des faux négatifs de votre liste :
faux_exemples = [
    21985, 10285, 47191, 23141, 30759, 1599, 1140923, 2845, 30981, 
    97017, 21509, 30621, 57485, 70845, 168291, 24399, 46241, 59455, 
    2079, 13439, 378383, 11911, 97763, 116051, 15531, 27571, 2219, 
    21309, 29267, 14733, 1749, 1981345, 110015, 90147
]

print("Analyse de la factorisation des faux positifs :")
for n in faux_exemples:
    factors = factorint(n)
    # Affichage sous forme "facteur^puissance * ..."
    fac_str = " * ".join([f"{k}^{v}" if v > 1 else f"{k}" for k, v in factors.items()])
    print(f"{n:8d} = {fac_str}")