import torch
import torch.nn as nn
from torch.nn import functional as F
import re

# --- 0. FONCTION DE GÉNÉRATION DU CORPUS (pour le vocabulaire) ---

BASE_TOKENS = [
    (2, "10", "01"),
    (3, "11", "00"),
    (5, "101", "010"),
    (7, "111", "000"),
    (17, "10001", "01110"),
    (19, "10011", "01100"),
    (409, "110011001", "001100110")
]

VALID_BRICKS = set()
for val, b1, b2 in BASE_TOKENS:
    VALID_BRICKS.add(b1)
    VALID_BRICKS.add(b2)

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def decompose_number(n_bin):
    if n_bin in VALID_BRICKS:
        return [n_bin]
    max_len = max(len(b) for b in VALID_BRICKS)
    for i in range(min(len(n_bin), max_len), 0, -1):
        prefix = n_bin[:i]
        if prefix in VALID_BRICKS:
            suffix = n_bin[i:]
            if not suffix:
                return [prefix]
            sub_res = decompose_number(suffix)
            if sub_res is not None:
                return [prefix] + sub_res
    return None

def get_tag_by_length(n):
    mapping = {2: ('A', 'a'), 3: ('B', 'b'), 4: ('C', 'c'), 5: ('D', 'd'), 6: ('E', 'e')}
    return mapping.get(n, (None, None))

def get_modulation(p):
    r6 = p % 6
    mod6_str = "+" if r6 == 1 else ("-" if r6 == 5 else "&")
    r4 = p % 4
    mod4_str = "*" if r4 == 1 else ("%" if r4 == 3 else "&")
    return f"{mod6_str}, {mod4_str}"

def generate_pure_syntax_corpus(max_primes=800):
    primes = []
    c = 2
    while len(primes) < max_primes:
        if is_prime(c):
            primes.append(c)
        c += 1

    prime_data = []
    global_tokens = []

    for p in primes:
        p_bin = bin(p)[2:]
        bricks = decompose_number(p_bin)
        prime_data.append({'p': p, 'mod_str': get_modulation(p), 'bricks': bricks})
        if bricks:
            for b in bricks:
                global_tokens.append({'token': b})

    open_tags = [[] for _ in range(len(global_tokens))]
    close_tags = [[] for _ in range(len(global_tokens))]

    for N in range(2, 7):
        for i in range(len(global_tokens) - N + 1):
            window = global_tokens[i:i+N]
            concat_bin = "".join([t['token'] for t in window])
            try:
                val = int(concat_bin, 2)
                if is_prime(val):
                    open_char, close_char = get_tag_by_length(N)
                    if open_char and open_char not in open_tags[i]:
                        open_tags[i].append(open_char)
                    if close_char and close_char not in close_tags[i + N - 1]:
                        close_tags[i + N - 1].append(close_char)
            except ValueError:
                pass

    syntax_blocks = []
    token_global_idx = 0
    for item in prime_data:
        mod_str = item['mod_str']
        bricks = item['bricks']
        if bricks is not None:
            brick_strs = []
            for b in bricks:
                o_str = "".join(open_tags[token_global_idx])
                c_str = "".join(close_tags[token_global_idx])
                token_global_idx += 1
                brick_strs.append(f"{o_str}({b}){c_str}")
            bricks_str = "".join(brick_strs)
        else:
            bricks_str = "(néant)"
        syntax_blocks.append(f"<[{mod_str}]{bricks_str}>")
    return "".join(syntax_blocks)

# --- 1. ARCHITECTURE MINI-GPT ---

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
        return F.softmax(wei, dim=-1) @ self.value(x)

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

# --- 2. AUDIT ET NOTATION ---

def analyze_generation():
    device = 'cpu'
    
    corpus_text = generate_pure_syntax_corpus(max_primes=800)
    chars = sorted(list(set(corpus_text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '') for i in l])

    model = GigogneLanguageModel(vocab_size, n_embd=128, block_size=64, n_head=4, n_layer=4).to(device)
    model.load_state_dict(torch.load("prime_gigogne_model.pth", map_location=device))
    model.eval()

    context = torch.tensor(encode("<[&, &]AC(10)><"), dtype=torch.long, device=device).unsqueeze(0)
    generated = decode(model.generate(context, max_new_tokens=150)[0].tolist())

    print("\n" + "="*60)
    print("TEXTE GÉNÉRÉ PAR LE MODÈLE :")
    print(generated)
    print("="*60 + "\n")

    # Compteurs pour la notation
    total_score = 0
    max_possible_score = 0

    # 1. Analyse des blocs
# 1. Analyse des blocs (version robuste)
    blocks = re.findall(r'<\[(.*?)\](.*?)>', generated)
    print("--- 1. ANALYSE DES BLOCS ---")
    for idx, (mod_str, content) in enumerate(blocks):
        max_possible_score += 10
        
        bricks = re.findall(r'\((.*?)\)', content)
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
        expected_mod6 = "+" if r6 == 1 else ("-" if r6 == 5 else "&")
        expected_mod4 = "*" if r6 == 1 else ("%" if r6 == 3 else "&")
        
        mod_parts = [m.strip() for m in mod_str.split(',')]
        actual_mod6 = mod_parts[0] if len(mod_parts) > 0 else "&"
        actual_mod4 = mod_parts[1] if len(mod_parts) > 1 else "&"
        
        mod6_ok = (actual_mod6 == expected_mod6)
        mod4_ok = (actual_mod4 == expected_mod4)

        block_pts = 2
        if prime_status: block_pts += 5
        if mod6_ok and mod4_ok: block_pts += 3
        total_score += block_pts

        print(f"Bloc {idx+1} : Modulations [{mod_str}] | Briques : {clean_bricks} -> Décimal : {val_dec} [+{block_pts}/10 pts]")
        print(f"   -> Premier ? {'OUI 🟢 (+5)' if prime_status else 'NON 🔴 (0)'}")
        print(f"   -> Modulations valides ? {'OUI 🟢 (+3)' if (mod6_ok and mod4_ok) else 'NON 🔴 (0)'}\n")

    # 2. Inspection des balises lettres
# 2. Inspection des balises lettres (version robuste)
    print("--- 2. INSPECTION DES BALISES LETTRES ---")
    tag_lengths = {'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6}
    
    for uppercase, expected_len in tag_lengths.items():
        lowercase = uppercase.lower()
        up_pos = [m.start() for m in re.finditer(uppercase, generated)]
        for p_start in up_pos:
            p_end = generated.find(lowercase, p_start)
            if p_end != -1:
                max_possible_score += 10 # 4 taille exacte + 6 primalité transversale
                segment = generated[p_start:p_end+1]
                bricks_in_segment = re.findall(r'\((.*?)\)', segment)
                count_tokens = len(bricks_in_segment)
                
                # Nettoyage robuste pour ne garder que le binaire pur
                clean_segment_bricks = []
                for b in bricks_in_segment:
                    pure_b = "".join([c for c in b if c in '01'])
                    if pure_b:
                        clean_segment_bricks.append(pure_b)
                        
                concat_tag_bin = "".join(clean_segment_bricks)
                val_tag_dec = int(concat_tag_bin, 2) if concat_tag_bin else 0
                is_p = is_prime(val_tag_dec)
                
                tag_pts = 0
                size_ok = (count_tokens == expected_len)
                if size_ok: tag_pts += 4
                if size_ok and is_p: tag_pts += 6
                total_score += tag_pts

                print(f"Balise [{uppercase} ... {lowercase}] : [+{tag_pts}/10 pts]")
                print(f"   - Segment : {segment}")
                print(f"   - Briques trouvées ({count_tokens} tokens) | Attendu : {expected_len}")
                if size_ok:
                    print(f"   - Taille validée ! (+4) Concaténation = {val_tag_dec} -> Premier ? {'OUI 🟢 (+6)' if is_p else 'NON 🔴 (0)'}")
                else:
                    print(f"   - ⚠️ Écart de taille (0 pt).")
                print("-" * 40)

    # Note Finale
    print("\n" + "="*60)
    final_percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
    print(f"🎯 SCORE DE GÉNÉRATION DU MODÈLE : {total_score} / {max_possible_score} ({final_percentage:.1f}%)")
    print("="*60 + "\n")

if __name__ == "__main__":
    analyze_generation()