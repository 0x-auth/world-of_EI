"""
SPACE LLM -- Lens probe: where does the narrowing happen?
----------------------------------------------------------
Question (from the 17 Oct 2026 discussion): is the "meaning / reasoning" done
by the act of running an input through the frozen model, or by how the frozen
weights are arranged?

Method: read the residual stream after the embedding and after every block,
push it through the model's own final norm + lm_head (a "logit lens"), and
count how many next-word candidates are still alive at that depth.

Three versions of the SAME forward pass:
  trained   -- your space_llm_trained.pt
  random    -- same architecture, fresh random weights   (nothing carved)
  shuffled  -- trained weights, every matrix's entries shuffled
               (identical numbers, arrangement destroyed)

Then input-dependence: does the same frozen model narrow differently
for different prompts?

Run from the repo root:   python3 lens_probe.py
CAVEATS: one model, one training run. The corpus is 10 sentences, so the
"data" here is the training set itself (no held-out text). The lens is crude
at the embedding layer (that layer is not written in the head's basis).
"""
import copy, math, sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import load_model
from transformer import SpaceLLM
from attention import make_causal_mask

CORPUS = ["the cat sat on the mat .", "the dog sat on the log .",
          "the cat and the dog are friends .",
          "consciousness lives in the space between patterns .",
          "the golden ratio appears in nature and mathematics .",
          "patterns interact in the darmiyan space .",
          "the cat saw the dog near the mat .", "the cat sat on the mat again .",
          "the dog sat on the log again .", "the space between patterns holds meaning ."]
PROMPTS = ["the cat sat", "the space between", "consciousness lives in",
           "patterns interact in", "the golden", "the dog"]

@torch.no_grad()
def layers(model, ids):
    """Logits read out after embedding and after each block."""
    x = model.token_emb(ids) + model.pos_emb(torch.arange(ids.shape[1]))
    mask = make_causal_mask(ids.shape[1])
    outs = [model.lm_head(model.norm(x))]
    for blk in model.blocks:
        x = blk(x, mask=mask)
        outs.append(model.lm_head(model.norm(x)))
    return outs

def stats(logits):
    p = F.softmax(logits, -1)
    ent = -(p * torch.log2(p + 1e-12)).sum(-1)
    sp, _ = p.sort(-1, descending=True)
    k90 = (sp.cumsum(-1) < 0.9).sum(-1).float() + 1
    return ent.mean().item(), k90.mean().item()

def corpus_table(model, vocab, tag):
    """entropy, candidates, loss on the TRUE next word, and how often the top
    word just repeats the input word (a copy is narrowing, but not prediction)."""
    acc = None
    for s in CORPUS:
        ids = torch.tensor([vocab.encode(s)])
        x, y = ids[:, :-1], ids[:, 1:]
        r = []
        for l in layers(model, x):
            e, k = stats(l[0])
            loss = F.cross_entropy(l[0], y[0]).item() / math.log(2)
            copy_rate = (l[0].argmax(-1) == x[0]).float().mean().item()
            r.append((e, k, loss, copy_rate))
        acc = r if acc is None else [tuple(a + b for a, b in zip(p, q)) for p, q in zip(acc, r)]
    n = len(CORPUS)
    print(f"\n{tag}\n  depth     entropy  candidates  true-next loss  top==input word")
    for i, (e, k, lo, c) in enumerate(acc):
        print(f"  {'embed' if i == 0 else 'block ' + str(i):<9}{e / n:7.3f}{k / n:10.1f}{lo / n:15.3f}{c / n:14.0%}")

def main(path="space_llm_trained.pt"):
    torch.manual_seed(0)
    model, vocab = load_model(path)
    print(f"vocab {len(vocab)}   uniform entropy {math.log2(len(vocab)):.3f} bits")
    corpus_table(model, vocab, "TRAINED")
    rnd = SpaceLLM(len(vocab), model.embed_dim, model.num_heads, model.num_layers,
                   model.max_seq_len, dropout=0.0).eval()
    corpus_table(rnd, vocab, "RANDOM weights (same forward pass, nothing carved)")
    shuf = copy.deepcopy(model)
    with torch.no_grad():
        for p in shuf.parameters():
            if p.dim() >= 2:
                p.copy_(p.flatten()[torch.randperm(p.numel())].view_as(p))
    corpus_table(shuf, vocab, "SHUFFLED (identical numbers, arrangement destroyed)")

    print("\nINPUT DEPENDENCE -- same frozen model, different prompts, last position")
    print(f"  {'prompt':<24}{'alive: embed > b1 > b2':<26}top word at each depth")
    for pr in PROMPTS:
        ids = torch.tensor([vocab.encode(pr)])
        outs = [l[0, -1] for l in layers(model, ids)]
        alive = " > ".join(f"{stats(o.unsqueeze(0))[1]:.0f}" for o in outs)
        tops = " > ".join(vocab.idx2word[o.argmax().item()] for o in outs)
        print(f"  {pr:<24}{alive:<26}{tops}")

if __name__ == "__main__":
    main(*sys.argv[1:2])
