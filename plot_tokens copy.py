import matplotlib.pyplot as plt
import numpy as np

# Exemple de données : pour chaque nombre 'n', une liste de tokens associés
# (Remplacez cette liste par vos propres données extraites de votre corpus/génération)
# Format : (n, [('prime', valeur) ou ('barre', valeur)])
donnees_exemple = [
    (3, [('prime', 2), ('barre', 2), ('prime', 3)]),  # S = 2 - 2 + 3 = 3
    (5, [('prime', 5)]),                             # S = 5
    (7, [('prime', 3), ('prime', 4)]),               # S = 3 + 4 = 7
    (11, [('prime', 6), ('barre', 2), ('prime', 7)]), # S = 6 - 2 + 7 = 11
    (13, [('prime', 13)]),                           # S = 13
    (17, [('prime', 10), ('barre', 3), ('prime', 10)]),# S = 10 - 3 + 10 = 17
]

x_coords = []
y_coords = []
valeurs_n = []

for n, tokens in donnees_exemple:
    # Calcul de S selon vos règles
    S = 0
    for type_token, val in tokens:
        if type_token == 'prime':
            S += val
        elif type_token == 'barre':
            S -= val
            
    # Calcul des coordonnées polaires transformées en cartésiennes
    x = n * np.cos(S)
    y = n * np.sin(S)
    
    x_coords.append(x)
    y_coords.append(y)
    valeurs_n.append(n)

# Tracé du graphique
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, c=valeurs_n, cmap='viridis', s=100, edgecolors='k')
plt.colorbar(label="Valeur de n")

# Ajout des étiquettes pour voir quel point correspond à quel n
for i, txt in enumerate(valeurs_n):
    plt.annotate(f"n={txt} (S={txt})", (x_coords[i], y_coords[i]), fontsize=9, xytext=(5, 5), textcoords='offset points')

plt.axhline(0, color='grey', linewidth=0.5, linestyle='--')
plt.axvline(0, color='grey', linewidth=0.5, linestyle='--')
plt.title("Visualisation géométrique (n * cos(S), n * sin(S))")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()