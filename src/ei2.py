#!/usr/bin/env python3
"""
EI v2 — Emergent Intelligence with a meaning-space substrate.

Change from v1: v1 predicted from LITERAL token transitions (a bigram counter).
It could not handle unseen material because "unknown" meant "exact string never seen."
v2 represents every context as a VECTOR (hashed char-n-grams). Prediction for an
unseen context falls back to the CONSEQUENCE OF THE NEAREST KNOWN CONTEXT in meaning
space. "Unknown" now means "nothing NEAR this," a far weaker and more useful failure.
This is the 'connect across' mechanism: relatedness, not identity.

Still EI, not AI: it learns AFTER genesis, from what it observes, persisting to a tape.
The interpreter (Python) remains the external constructor. No weights are pretrained.
"""
from __future__ import annotations
import argparse, json, re, math
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")
DIM = 512  # embedding dimension (hashed feature space)

def _hash(tok: str) -> int:
    h = 1469598103934665603
    for ch in tok.encode("utf-8"):
        h ^= ch; h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def embed(text: str) -> np.ndarray:
    """Self-contained embedding: hashed char 3/4-grams + whole tokens. No external model."""
    v = np.zeros(DIM, dtype=np.float64)
    toks = [w.lower() for w in WORD_RE.findall(text)]
    if not toks: return v
    for t in toks:
        s = "#" + t + "#"
        feats = [t] + [s[i:i+3] for i in range(len(s)-2)] + [s[i:i+4] for i in range(len(s)-3)]
        for f in feats:
            idx = _hash(f) % DIM
            sign = 1.0 if (_hash("s"+f) & 1) else -1.0
            v[idx] += sign
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

class EI2:
    def __init__(self, root: Path):
        self.root = root
        self.tape_path = root/"tape"/"self.json"
        self.program_path = root/"programs"/"transition_reasoner.json"
        self.results_path = root/"results"/"learning_curve.json"
        for p in (self.tape_path, self.program_path, self.results_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        # active model: context-string -> Counter(consequence)
        self.model: dict[str, Counter] = defaultdict(Counter)
        # meaning-space memory, rebuilt from tape (reconstruction, not retrieval)
        self.ctx_list: list[str] = []
        self.ctx_mat: np.ndarray | None = None
        self.scars: list[dict] = []
        self.observations = self.unknowns = self.errors = self.correct = 0
        self._load_tape()

    def _load_tape(self):
        if not self.tape_path.exists(): return
        try: tape = json.loads(self.tape_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError: return
        for item in tape.get("knowledge", []):
            c, q, e = item.get("context"), item.get("consequence"), int(item.get("evidence",1))
            if c and q: self.model[c][q] += max(e,1)
        self.scars = tape.get("scars", [])
        self._rebuild_space()

    def _rebuild_space(self):
        """Reconstruct the meaning-space index from the active model. This is the
        'memory as reconstruction' step: the vector index is not stored, it is rebuilt."""
        self.ctx_list = list(self.model.keys())
        if self.ctx_list:
            self.ctx_mat = np.vstack([embed(c) for c in self.ctx_list])
        else:
            self.ctx_mat = None

    @staticmethod
    def lines(text): return [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    @staticmethod
    def tokens(line): return [w.lower() for w in WORD_RE.findall(line)]
    def read_world(self, path):
        if not path.exists(): raise FileNotFoundError(path)
        return self.lines(path.read_text(encoding="utf-8", errors="ignore"))
    def contexts(self, line):
        w = self.tokens(line); pairs=[]
        for i in range(len(w)-1): pairs.append((w[i], w[i+1]))
        for i in range(len(w)-2): pairs.append((w[i]+" "+w[i+1], w[i+2]))
        return pairs

    def predict(self, context, threshold=0.45):
        """Exact if known; else nearest-in-meaning-space above threshold; else None."""
        exact = self.model.get(context)
        if exact: return exact.most_common(1)[0][0], "exact", 1.0
        if self.ctx_mat is None or not self.ctx_list: return None, "unknown", 0.0
        q = embed(context)
        if not q.any(): return None, "unknown", 0.0
        sims = self.ctx_mat @ q
        j = int(np.argmax(sims)); best = float(sims[j])
        if best >= threshold:
            return self.model[self.ctx_list[j]].most_common(1)[0][0], "analogy", best
        return None, "unknown", best

    def evaluate(self, lines, threshold=0.45):
        total=correct=unknown=analogy_hits=analogy_tries=0
        for line in lines:
            for context, actual in self.contexts(line):
                pred, kind, sim = self.predict(context, threshold)
                total += 1
                if pred is None: unknown += 1
                else:
                    if kind=="analogy":
                        analogy_tries += 1
                        if pred==actual: analogy_hits += 1
                    if pred==actual: correct += 1
        return {"relations":total, "correct":correct, "unknown":unknown,
                "accuracy": correct/total if total else 0.0,
                "analogy_tries":analogy_tries, "analogy_hits":analogy_hits}

    def observe(self, line, learn=True, threshold=0.45):
        for context, actual in self.contexts(line):
            pred, kind, sim = self.predict(context, threshold)
            self.observations += 1
            if pred is None:
                self.unknowns += 1
                self.scars.append({"context":context,"predicted":None,"observed":actual,"kind":"first_encounter"})
            elif pred == actual:
                self.correct += 1
            else:
                self.errors += 1
                self.scars.append({"context":context,"predicted":pred,"observed":actual,
                                   "kind":"analogy_miss" if kind=="analogy" else "prediction_mismatch","sim":round(sim,3)})
            if learn: self.model[context][actual] += 1

    def train(self, lines, threshold=0.45):
        for line in lines: self.observe(line, learn=True, threshold=threshold)
        self._rebuild_space()
        return {"lines":len(lines), "knowledge_relations":len(self.model), "scars":len(self.scars)}

    def persist(self):
        knowledge=[]
        for context,counts in sorted(self.model.items()):
            q,e = counts.most_common(1)[0]
            knowledge.append({"context":context,"consequence":q,"evidence":e})
        tape={"version":2,"machine":"EI","substrate":"hashed-char-ngram-embedding","dim":DIM,
              "knowledge":knowledge[:4000],"scars":self.scars[-2000:],
              "metrics":{"observations":self.observations,"unknowns":self.unknowns,
                         "errors":self.errors,"correct":self.correct,
                         "knowledge_relations":len(knowledge),"scar_count":len(self.scars)}}
        self.tape_path.write_text(json.dumps(tape,indent=2),encoding="utf-8")
        rules=[{"if_context":c,"predict":self.model[c].most_common(1)[0][0],
                "evidence":self.model[c].most_common(1)[0][1]} for c in sorted(self.model)]
        self.program_path.write_text(json.dumps(
            {"name":"transition_reasoner","language":"EI-safe-dsl-v2",
             "prediction":"exact-or-nearest-in-meaning-space","rules":rules[:1000]},indent=2),encoding="utf-8")


def demo():
    root = Path(__file__).resolve().parent.parent
    # fresh tape for a clean comparison
    (root/"tape"/"self.json").unlink(missing_ok=True)
    ei = EI2(root)
    world = ei.read_world(root/"input"/"world.txt")
    ei.train(world)
    # UNSEEN but related file (same relational vocabulary, new pairings)
    unseen = ei.read_world(root/"input"/"philosophy.md")
    genuinely_new = root/"input"/"unseen.md"
    genuinely_new.write_text(
        "learning reduces error\nerror changes prediction\nprediction improves action\n"
        "recovery preserves identity\nmemory biases transition\nknowledge enables recovery\n",
        encoding="utf-8")
    new = ei.read_world(genuinely_new)

    print("="*64); print("EI v2 — meaning-space substrate"); print("="*64)
    print(f"trained on world.txt: {len(world)} lines, {len(ei.model)} known contexts\n")
    for name, corpus in [("held-related (philosophy.md)", unseen), ("genuinely-unseen (new)", new)]:
        # bigram baseline = threshold 1.1 (impossible), i.e. exact-only
        base = ei.evaluate(corpus, threshold=1.1)
        emb  = ei.evaluate(corpus, threshold=0.45)
        print(f"[{name}]")
        print(f"  bigram (exact-only)   acc={base['accuracy']:.3f}  unknown={base['unknown']}/{base['relations']}")
        print(f"  EI v2 (meaning-space) acc={emb['accuracy']:.3f}  unknown={emb['unknown']}/{emb['relations']}"
              f"  analogy {emb['analogy_hits']}/{emb['analogy_tries']}")
        print()
    ei.persist()
    print("passes README test if EI v2 acc > bigram acc on unseen material.")

if __name__ == "__main__":
    demo()
