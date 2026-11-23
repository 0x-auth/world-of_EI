"""organism run | strike | keep | status | clean"""
import argparse, json, os, sys, hashlib
import numpy as np
from . import __version__, core, sources, seed as seedmod
from .store import Store, now


def cmd_run(a):
    print("reading sources")
    ev, meta = sources.merge(sources.open_source(s) for s in a.sources)
    excl = [x.strip() for x in a.exclude.split(",") if x.strip()]
    chats = core.threads_from_events(ev, meta, a.human_only, a.since, excl)
    if a.gap:
        chats = core.split_gaps(chats, a.gap)
    store = Store(a.store)
    struck = store.struck_pairs()
    focus = None
    if a.focus:
        want = dict((k.strip(), int(v)) for k, v in (p.split("=") for p in a.focus.split(",")))
        focus = set()
        for acct_key, n in want.items():
            mine = sorted([i for i, c in enumerate(chats) if acct_key in c["acct"]],
                          key=lambda i: chats[i]["updated"], reverse=True)[:n]
            focus |= set(mine)
    vocab, common = core.build_vocab(chats, a.vocab, a.min_df, a.max_df)
    print(f"left out {len(common)} terms found in over {a.max_df:.0%} of threads")
    print(f"{len(chats)} threads, {sum(len(c['msgs']) for c in chats)} events, vocabulary {len(vocab)}, "
          f"{len(struck)} struck term pairs")

    born_f = born_b = None; seed_from = None
    seed_path = a.seed
    if seed_path == "last":
        lr = store.last_run(); seed_path = lr[1] if lr else None
    if seed_path and seed_path != "none":
        born_f, born_b, kept = seedmod.load(seed_path, vocab, struck)
        seed_from = seed_path
        print(f"carried seed {seed_path}: {kept}/{len(vocab)} words remembered")
    else:
        print("newborn (no seed)")

    org, n, b0, b1 = core.live(chats, vocab, a.alpha, born=born_f)
    print(f"forward: {n} predictions; surprise first {b0:.2f} bits, last {b1:.2f} bits")
    org_b, c0, c1 = None, None, None
    if not a.single:
        org_b, nb, c0, c1 = core.live(chats, vocab, a.alpha, reverse=True, born=born_b)
        print(f"retrocausal: {nb} predictions; surprise first {c0:.2f} bits, last {c1:.2f} bits")

    L = core.link_matrix(org.W, org_b.W if org_b else None)
    applied = core.apply_strikes(L, vocab, struck)
    if a.parts:
        links = core.part_links(chats, vocab, L, size=a.parts, top=a.top, focus=focus, min_msgs=a.min_msgs)
    else:
        links = core.chat_links(chats, vocab, L, top=a.top, focus=focus, min_msgs=a.min_msgs)

    run_id = now()
    os.makedirs(a.out, exist_ok=True)
    seed_file = os.path.abspath(os.path.join(a.out, f"seed_{run_id}.npz"))
    seedmod.save(seed_file, vocab, org, org_b)
    params = {k: v for k, v in vars(a).items() if k != "func"}
    store.save_run({"run_id": run_id, "created_at": run_id, "threads": len(chats),
                    "events": sum(len(c["msgs"]) for c in chats), "vocab": len(vocab),
                    "bits_first": b0, "bits_last": b1, "retro_first": c0, "retro_last": c1,
                    "seed_from": seed_from, "seed_file": seed_file, "strikes_applied": applied,
                    "params": {**params, "version": __version__}})
    store.save_links(run_id, links)
    json.dump(links, open(f"{a.out}/links.json", "w"), indent=1, ensure_ascii=False)
    with open(f"{a.out}/links.md", "w") as f:
        f.write(f"# Hidden links · run {run_id} ({len(links)})\n"
                f"Strike what's wrong: `organism strike {run_id} <rank>` · keep: `organism keep {run_id} <rank>`\n\n")
        for i, l in enumerate(links, 1):
            tag = "cross-account" if l["cross_account"] else "same account"
            ra = f" msgs {l['a_msgs'][0]}–{l['a_msgs'][1]}" if "a_msgs" in l else ""
            rb = f" msgs {l['b_msgs'][0]}–{l['b_msgs'][1]}" if "b_msgs" in l else ""
            f.write(f"{i}. **{l['a_title']}**{ra} ({l['a_acct']}) ↔ **{l['b_title']}**{rb} ({l['b_acct']}) · {tag}\n"
                    f"   map {l['map']:.4f} · shared words {l['words']:.2f} · carried by "
                    + ", ".join(f"{u}→{v}" for u, v, _ in l["carried_by"]) + "\n\n")
    print(f"run {run_id}: {len(links)} links → {a.out}/links.md · store {a.store} · seed {seed_file}")


def cmd_strike(a):
    s = Store(a.store)
    if a.pair:
        s.strike_pair(a.pair[0], a.pair[1], reason=a.reason); print(f"struck {a.pair[0]}↔{a.pair[1]}")
    else:
        for r in a.rank:
            print(f"struck {a.run} #{r}: " + ", ".join(f"{u}↔{v}" for u, v in s.strike(a.run, r, a.reason)))


def cmd_keep(a):
    s = Store(a.store)
    for r in a.rank: s.keep(a.run, r)
    print(f"kept {a.run} " + " ".join(f"#{r}" for r in a.rank))


def cmd_status(a):
    s = Store(a.store)
    runs = s.q("SELECT run_id, threads, vocab, bits_first, bits_last, seed_from IS NOT NULL, strikes_applied "
               "FROM organism_runs ORDER BY run_id")
    for r in runs:
        st = dict(s.q("SELECT status, COUNT(*) FROM organism_links WHERE run_id=? GROUP BY status", (r[0],)))
        print(f"{r[0]}  threads {r[1]}  vocab {r[2]}  bits {r[3]:.2f}→{r[4]:.2f}  "
              f"{'seeded' if r[5] else 'newborn'}  strikes used {r[6]}  links {st}")
    print(f"struck term pairs: {len(s.struck_pairs())}")
    if runs and a.top:
        for row in s.q("SELECT rank, title_a, title_b, status FROM organism_links WHERE run_id=? AND rank<=? "
                       "ORDER BY rank", (runs[-1][0], a.top)):
            print(f"  #{row[0]:<3} [{row[3]}] {row[1]} ↔ {row[2]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="organism", description="EI organism: seed → medium → rule → filter → seed")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(required=True)
    st = dict(default=os.environ.get("ORGANISM_STORE", "organism_out/organism.db"),
              help="sqlite path | d1:<uuid> | postgres:DSN (env ORGANISM_STORE)")

    r = sub.add_parser("run", help="live once over the sources, write links, pass the seed on")
    r.add_argument("sources", nargs="+", help="export folder | sqlite:file.db | d1:<uuid> | postgres:DSN")
    r.add_argument("--store", **st)
    r.add_argument("--seed", default="last", help="'last' (default) | 'none' (newborn) | path to seed .npz")
    r.add_argument("--out", default="organism_out")
    r.add_argument("--human-only", action="store_true")
    r.add_argument("--vocab", type=int, default=4000)
    r.add_argument("--min-df", type=int, default=3)
    r.add_argument("--max-df", type=float, default=0.3)
    r.add_argument("--min-msgs", type=int, default=6)
    r.add_argument("--alpha", type=float, default=0.5)
    r.add_argument("--top", type=int, default=60)
    r.add_argument("--single", action="store_true")
    r.add_argument("--gap", type=float, default=0, help="split threads into sessions at gaps over this many hours (e.g. 6 for WhatsApp); 0 = off")
    r.add_argument("--parts", type=int, default=20, help="link parts of N messages (default 20); 0 = whole threads, as v1.4")
    r.add_argument("--since", default=None)
    r.add_argument("--exclude", default="Cross-account chat synthesis")
    r.add_argument("--focus", default="", help="e.g. localhost=10,phiinverse=10 ; '' = all threads")
    r.set_defaults(func=cmd_run)

    k = sub.add_parser("strike", help="mark a link wrong; its term pairs are cut from every later life")
    k.add_argument("run", nargs="?"); k.add_argument("rank", nargs="*", type=int)
    k.add_argument("--pair", nargs=2, metavar=("TERM_A", "TERM_B"))
    k.add_argument("--reason", default=""); k.add_argument("--store", **st)
    k.set_defaults(func=cmd_strike)

    kp = sub.add_parser("keep", help="mark a link right")
    kp.add_argument("run"); kp.add_argument("rank", nargs="+", type=int); kp.add_argument("--store", **st)
    kp.set_defaults(func=cmd_keep)

    s = sub.add_parser("status", help="runs, strikes, top links of the last run")
    s.add_argument("--store", **st); s.add_argument("--top", type=int, default=10)
    s.set_defaults(func=cmd_status)

    c = sub.add_parser("clean", help="clean raw Claude exports (one folder per account) into plain threads")
    c.add_argument("src"); c.add_argument("out")
    c.add_argument("--exclude", default="Cross-account chat synthesis")
    c.set_defaults(func=lambda a: _clean(a))

    a = ap.parse_args(argv)
    a.func(a)


def _clean(a):
    from . import clean
    sys.argv = ["organism clean", a.src, a.out, "--exclude", a.exclude]
    clean.main()


if __name__ == "__main__":
    main()
