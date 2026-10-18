import torch
import torch.nn as nn
from torch.nn import functional as F
import re
from verify_model import generate_pure_syntax_corpus

# --- 1. FONCTIONS DE NOTATION ROBUSTE ---

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
    score = 0
    max_score = 0
    
    # Analyse des blocs
    blocks = re.findall(r'<\[(.*?)\](.*?)>', generated_text)
    for mod_str, content in blocks:
        max_score += 10
        bricks = re.findall(r'\((.*?)\)', content)
        clean_bricks = [ "".join([c for c in b if c in '01']) for b in bricks ]
        clean_bricks = [b for b in clean_bricks if b]
                
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

    # Analyse des balises
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
                
                clean_segment_bricks = [ "".join([c for c in b if c in '01']) for b in bricks_in_segment ]
                clean_segment_bricks = [b for b in clean_segment_bricks if b]
                        
                concat_tag_bin = "".join(clean_segment_bricks)
                val_tag_dec = int(concat_tag_bin, 2) if concat_tag_bin else 0
                
                tag_pts = 0
                size_ok = (count_tokens == expected_len)
                if size_ok: tag_pts += 4
                if size_ok and is_prime(val_tag_dec): tag_pts += 6
                score += tag_pts

    return (score / max_score * 100) if max_score > 0 else 0

# --- 2. ARCHITECTURE MINI-GPT ---

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
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx

# --- 3. BOUCLE D'OPTIMISATION AUTOMATIQUE ---

def run_automated_loops(num_cycles=5, batch_size=40):
    device = 'cpu'
    
    corpus_text = generate_pure_syntax_corpus(max_primes=800)
    chars = sorted(list(set(corpus_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    model = GigogneLanguageModel(len(chars), n_embd=128, block_size=64, n_head=4, n_layer=4).to(device)
    model.load_state_dict(torch.load("prime_gigogne_model.pth", map_location=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(f"\n🚀 Démarrage de la boucle d'optimisation automatique ({num_cycles} cycles)...")
    
    for cycle in range(1, num_cycles + 1):
        print(f"\n--- CYCLE {cycle} / {num_cycles} ---")
        model.eval()
        
        # 1. Génération et notation d'un lot
        context = torch.tensor(encode("<[&, &]AC(10)><"), dtype=torch.long, device=device).unsqueeze(0)
        scored_generations = []
        
        for _ in range(batch_size):
            gen = decode(model.generate(context, max_new_tokens=100)[0].tolist())
            score = score_sequence(gen)
            scored_generations.append((score, gen))
            
        # Tri par score décroissant
        scored_generations.sort(key=lambda x: x[0], reverse=True)
        avg_score = sum(s[0] for s in scored_generations) / batch_size
        max_score_in_batch = scored_generations[0][0]
        
        print(f"   📊 Score moyen du lot : {avg_score:.1f}% | Meilleur score du lot : {max_score_in_batch:.1f}%")
        
        # Sélection des "élites" (le top 30% du lot)
        num_elite = max(5, int(batch_size * 0.3))
        elite_texts = [item[1] for item in scored_generations[:num_elite]]
        
        # 2. Fine-tuning sur les élites
        model.train()
        for epoch in range(40): # Petites passes d'apprentissage par cycle
            for text in elite_texts:
                tokens = encode(text)
                if len(tokens) > 65:
                    seq = torch.tensor(tokens[:65], dtype=torch.long, device=device).unsqueeze(0)
                    x, y = seq[:, :-1], seq[:, 1:]
                    _, loss = model(x, y)
                    
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        # Sauvegarde intermédiaire
        torch.save(model.state_dict(), "prime_gigogne_model.pth")
        print(f"   💾 Modèle optimisé et sauvegardé (Cycle {cycle} terminé).")

    print("\n✨ Optimisation automatique terminée avec succès ! Vous pouvez relancer 'py verify_model.py' pour voir l'évolution.")

if __name__ == "__main__":
    run_automated_loops(num_cycles=5, batch_size=40)