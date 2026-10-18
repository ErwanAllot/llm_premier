import torch
import torch.nn as nn
from torch.nn import functional as F
import re

# (On reprend les briques de base et les fonctions de vérification)
BASE_TOKENS = [
    (2, "10", "01"), (3, "11", "00"), (5, "101", "010"),
    (7, "111", "000"), (17, "10001", "01110"), (19, "10011", "01100"),
    (409, "110011001", "001100110")
]
VALID_BRICKS = {b for _, b1, b2 in BASE_TOKENS for b in (b1, b2)}

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def score_sequence(generated_text):
    """Calcule le score d'une séquence générée selon notre barème avec nettoyage robuste"""
    score = 0
    max_score = 0
    
    # 1. Analyse des blocs
    blocks = re.findall(r'<\[(.*?)\](.*?)>', generated_text)
    for mod_str, content in blocks:
        max_score += 10
        bricks = re.findall(r'\((.*?)\)', content)
        
        # Nettoyage robuste : on ne garde que les '0' et '1' dans chaque brique
        clean_bricks = []
        for b in bricks:
            pure_b = "".join([c for c in b if c in '01'])
            if pure_b:
                clean_bricks.append(pure_b)
                
        concat_bin = "".join(clean_bricks)
        val_dec = int(concat_bin, 2) if concat_bin else 0
        prime_status = is_prime(val_dec)
        
        r6 = val_dec % 6 if val_dec > 0 else 0
        r4 = val_dec % 4 if val_dec > 0 else 0
        exp_mod6 = "+" if r6 == 1 else ("-" if r6 == 5 else "&")
        exp_mod4 = "*" if r4 == 1 else ("%" if r4 == 3 else "&")
        
        mod_parts = [m.strip() for m in mod_str.split(',')]
        act_mod6 = mod_parts[0] if len(mod_parts) > 0 else "&"
        act_mod4 = mod_parts[1] if len(mod_parts) > 1 else "&"
        
        block_pts = 2
        if prime_status: block_pts += 5
        if act_mod6 == exp_mod6 and act_mod4 == exp_mod4: block_pts += 3
        score += block_pts

    # 2. Analyse des balises
    tag_lengths = {'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6}
    for uppercase, expected_len in tag_lengths.items():
        lowercase = uppercase.lower()
        up_pos = [m.start() for m in re.finditer(uppercase, generated_text)]
        for p_start in up_pos:
            p_end = generated_text.find(lowercase, p_start)
            if p_end != -1:
                max_score += 10
                segment = generated_text[p_start:p_end+1]
                bricks_in_segment = re.findall(r'\((.*?)\)', segment)
                count_tokens = len(bricks_in_segment)
                
                clean_segment_bricks = []
                for b in bricks_in_segment:
                    pure_b = "".join([c for c in b if c in '01'])
                    if pure_b:
                        clean_segment_bricks.append(pure_b)
                        
                concat_tag_bin = "".join(clean_segment_bricks)
                val_tag_dec = int(concat_tag_bin, 2) if concat_tag_bin else 0
                
                tag_pts = 0
                size_ok = (count_tokens == expected_len)
                if size_ok: tag_pts += 4
                if size_ok and is_prime(val_tag_dec): tag_pts += 6
                score += tag_pts

    return (score / max_score * 100) if max_score > 0 else 0

# (Modèle Mini-GPT - même architecture)
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head, n_embd, block_size)
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        return x + self.ffwd(self.ln2(x + self.sa(self.ln1(x))))

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
        logits = self.lm_head(self.ln_f(self.blocks(tok_emb + pos_emb)))
        
        if targets is None:
            loss = None
        else:
            # Remplacement de .view(...) par .reshape(...) pour éviter l'erreur de mémoire contiguë
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

def main():
    device = 'cpu'
    
    # On importe la fonction de génération directement depuis verify_model.py
    from verify_model import generate_pure_syntax_corpus
    
    corpus_text = generate_pure_syntax_corpus(max_primes=800)
    chars = sorted(list(set(corpus_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    model = GigogneLanguageModel(len(chars), n_embd=128, block_size=64, n_head=4, n_layer=4).to(device)
    model.load_state_dict(torch.load("prime_gigogne_model.pth", map_location=device))
    model.eval()

    print("Phase 1 : Génération et notation d'un lot de propositions par le modèle...")
    elite_texts = []
    
    # On génère 50 propositions
    context = torch.tensor(encode("<[&, &]AC(10)><"), dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(50):
        gen = decode(model.generate(context, max_new_tokens=100)[0].tolist())
        score = score_sequence(gen)
        # Seuil d'élitisme : on ne garde que ce qui dépasse par exemple 15% ou 20% de score
        if score >= 15.0: 
            elite_texts.append(gen)

    print(f"-> {len(elite_texts)} séquences 'élites' sélectionnées sur 50.")
    if not elite_texts:
        print("Aucune séquence n'a atteint le seuil. Baissez le seuil ou relâchez les critères.")
        return

    # Phase 2 : Fine-tuning du modèle sur ces meilleures séquences
    print("Phase 2 : Ré-entraînement (Fine-tuning) du modèle sur ses meilleures réussites...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    elite_data = torch.tensor([encode(text) for text in elite_texts], dtype=torch.long, device=device)
    
    for epoch in range(100): # 100 micro-itérations de renforcement
        # On prend des chunks de taille block_size (64)
        if elite_data.shape[1] > 64:
            x = elite_data[:, :-1]
            y = elite_data[:, 1:]
            _, loss = model(x[:, :64], y[:, :64])
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # Sauvegarde du modèle amélioré
    torch.save(model.state_dict(), "prime_gigogne_model.pth")
    print("Modèle amélioré et sauvegardé avec succès dans 'prime_gigogne_model.pth' !")

if __name__ == "__main__":
    main()