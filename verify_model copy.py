import torch
import torch.nn as nn
from torch.nn import functional as F
import re

# (On reprend exactement la même classe du modèle pour pouvoir charger les poids)
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size)
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GigogneLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=idx.device))
        x = self.ln_f(self.blocks(tok_emb + pos_emb))
        logits = self.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def analyze_generation():
    device = 'cpu'
    
    # 1. On régénère exactement le même corpus pour avoir le même vocabulaire de 25 caractères
    # (On peut utiliser la fonction de génération du corpus source)
    corpus_text = generate_pure_syntax_corpus(max_primes=800) # Assurez-vous d'avoir la fonction generate_pure_syntax_corpus dans le fichier ou copiez-la
    chars = sorted(list(set(corpus_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    # Chargement du modèle déjà entraîné avec la bonne taille de vocabulaire
    model = GigogneLanguageModel(vocab_size, n_embd=128, block_size=64, n_head=4, n_layer=4).to(device)
    model.load_state_dict(torch.load("prime_gigogne_model.pth", map_location=device))
    model.eval()

    # Génération
    context = torch.tensor(encode("<[&, &]AC(10)><"), dtype=torch.long, device=device).unsqueeze(0)
    generated = decode(model.generate(context, max_new_tokens=150)[0].tolist())

    print("\n" + "="*60)
    print("TEXTE GÉNÉRÉ PAR LE MODÈLE :")
    print(generated)
    print("="*60 + "\n")

    # --- 1. ANALYSE DES BLOCS ET DES NOMBRES PREMIERS ---
    blocks = re.findall(r'<\[(.*?)\](.*?)>', generated)
    
    print("--- 1. ANALYSE DES BLOCS ---")
    flat_tokens_list = [] # Pour stocker les briques dans l'ordre global avec leurs index
    
    for idx, (mod_str, content) in enumerate(blocks):
        # Extraction des briques binaires à l'intérieur du bloc
        bricks = re.findall(r'\((.*?)\)', content)
        concat_bin = "".join(bricks)
        
        val_dec = int(concat_bin, 2) if concat_bin else 0
        prime_status = is_prime(val_dec)
        
        # Vérification des modulations
        r6 = val_dec % 6 if val_dec > 0 else 0
        r4 = val_dec % 4 if val_dec > 0 else 0
        
        expected_mod6 = "+" if r6 == 1 else ("-" if r6 == 5 else "&")
        expected_mod4 = "*" if r4 == 1 else ("%" if r4 == 3 else "&")
        
        mod_parts = [m.strip() for m in mod_str.split(',')]
        actual_mod6 = mod_parts[0] if len(mod_parts) > 0 else "&"
        actual_mod4 = mod_parts[1] if len(mod_parts) > 1 else "&"
        
        mod6_match = (actual_mod6 == expected_mod6)
        mod4_match = (actual_mod4 == expected_mod4)

        print(f"Bloc {idx+1} : Modulations [{mod_str}] | Briques : {bricks} -> Binaire concaténé : '{concat_bin}' (Décimal : {val_dec})")
        print(f"   -> Premier ? {'OUI 🟢' if prime_status else 'NON 🔴'}")
        print(f"   -> Mod 6 valide ? {mod6_match} (attendu: {expected_mod6}, trouvé: {actual_mod6})")
        print(f"   -> Mod 4 valide ? {mod4_match} (attendu: {expected_mod4}, trouvé: {actual_mod4})\n")

        # Enregistrement pour l'analyse des balises transversales
        for b in bricks:
            flat_tokens_list.append(b)

    # --- 2. ANALYSE DES BALISES LETTRES (TRANSVERSALES) ---
    print("--- 2. INSPECTION DES BALISES LETTRES ---")
    # Mapping des tailles de tokens attendues par lettre majuscule
    tag_lengths = {'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6}
    
    # On parcourt le texte pour chercher les paires de balises (ex: A ... a)
    # Dans la syntaxe, les lettres sont attachées aux briques, ex: AB(10)a
    for uppercase, expected_len in tag_lengths.items():
        lowercase = uppercase.lower()
        # Recherche de toutes les positions des majuscules dans le texte généré
        up_pos = [m.start() for m in re.finditer(uppercase, generated)]
        for p_start in up_pos:
            # Chercher la minuscule correspondante un peu plus loin
            p_end = generated.find(lowercase, p_start)
            if p_end != -1:
                # Extraire le sous-texte entre la majuscule et la minuscule
                segment = generated[p_start:p_end+1]
                # Extraire uniquement les briques (..)... dans ce segment
                bricks_in_segment = re.findall(r'\((.*?)\)', segment)
                count_tokens = len(bricks_in_segment)
                
                concat_tag_bin = "".join(bricks_in_segment)
                val_tag_dec = int(concat_tag_bin, 2) if concat_tag_bin else 0
                is_p = is_prime(val_tag_dec)
                
                print(f"Balise [{uppercase} ... {lowercase}] :")
                print(f"   - Segment brut : {segment}")
                print(f"   - Briques trouvées ({count_tokens} tokens) : {bricks_in_segment}")
                print(f"   - Attendu par la règle : {expected_len} tokens")
                
                if count_tokens == expected_len:
                    print(f"   - Taille validée ! Concaténation = {concat_tag_dec} -> Premier ? {'OUI 🟢' if is_p else 'NON 🔴'}")
                else:
                    print(f"   - ⚠️ Écart de taille : compte {count_tokens} tokens au lieu de {expected_len} (balise trop large ou coupée).")
                print("-" * 40)

if __name__ == "__main__":
    analyze_generation()