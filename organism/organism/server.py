"""The organism as a small HTTP service, for running next to its data (a Cloudflare container).
Stdlib only. One run at a time, in a background thread; the caller polls /status.

  GET  /health                 ok
  POST /reset                  forget the events and threads loaded so far
  POST /ingest                 {"events": [[thread_id, time, role, text], ...],
                                "threads": [[thread_id, title, account, updated], ...]}
  PUT  /seed                   the carried seed (.npz bytes); GET /seed returns the new one after a run
  POST /run                    {"strikes": [[a, b], ...], "gap": 6, "parts": 20, "since": null, "top": 200, ...}
  GET  /status                 {"state": "idle|running|done|error", "log": [...], "error": ...}
  GET  /result                 {"run": {...}, "links": [...]}
"""
import io, json, os, threading, traceback, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from . import __version__, core, seed as seedmod

STATE = {"events": [], "meta": {}, "seed_in": None, "seed_out": None, "state": "idle", "log": [],
         "error": None, "result": None}
LOCK = threading.Lock()


def log(msg):
    STATE["log"].append(f"{datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')} {msg}")
    STATE["log"] = STATE["log"][-200:]


def run(p):
    try:
        excl = p.get("exclude") or []
        chats = core.threads_from_events(STATE["events"], STATE["meta"], p.get("human_only", False), p.get("since"), excl)
        if p.get("gap"):
            chats = core.split_gaps(chats, float(p["gap"]))
        log(f"{len(chats)} threads, {sum(len(c['msgs']) for c in chats)} events")
        vocab, common = core.build_vocab(chats, p.get("vocab", 4000), p.get("min_df", 3), p.get("max_df", 0.3))
        log(f"vocabulary {len(vocab)}; left out {len(common)} common terms")
        struck = [tuple(x) for x in p.get("strikes", [])]
        born_f = born_b = None; kept = 0
        if STATE["seed_in"] is not None:
            path = "/tmp/seed_in.npz"; open(path, "wb").write(STATE["seed_in"])
            born_f, born_b, kept = seedmod.load(path, vocab, struck)
            log(f"carried seed: {kept}/{len(vocab)} words remembered")
        else:
            log("newborn (no seed)")
        alpha = p.get("alpha", 0.5)
        org, n, b0, b1 = core.live(chats, vocab, alpha, born=born_f)
        log(f"forward: {n} predictions, surprise {b0:.2f} -> {b1:.2f} bits")
        org_b, nb, c0, c1 = core.live(chats, vocab, alpha, reverse=True, born=born_b)
        log(f"retrocausal: {nb} predictions, surprise {c0:.2f} -> {c1:.2f} bits")
        L = core.link_matrix(org.W, org_b.W)
        applied = core.apply_strikes(L, vocab, struck)
        parts, top, mm = p.get("parts", 20), p.get("top", 200), p.get("min_msgs", 6)
        links = (core.part_links(chats, vocab, L, size=parts, top=top, min_msgs=mm) if parts
                 else core.chat_links(chats, vocab, L, top=top, min_msgs=mm))
        log(f"{len(links)} links")
        buf = io.BytesIO(); seedmod.save(buf, vocab, org, org_b); STATE["seed_out"] = buf.getvalue()
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        STATE["result"] = {"run": {"run_id": run_id, "threads": len(chats), "events": sum(len(c["msgs"]) for c in chats),
                                   "vocab": len(vocab), "bits_first": b0, "bits_last": b1, "retro_first": c0,
                                   "retro_last": c1, "seed_kept": kept, "seeded": STATE["seed_in"] is not None,
                                   "strikes_applied": applied, "params": {**{k: v for k, v in p.items() if k != "strikes"},
                                                                          "version": __version__}},
                           "links": links}
        STATE["state"] = "done"; log("done")
    except Exception as e:
        STATE["state"] = "error"; STATE["error"] = f"{e.__class__.__name__}: {e}"; log(traceback.format_exc()[-1500:])


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def _body(self):
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            out = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline(); break
                out += self.rfile.read(size); self.rfile.readline()
            return bytes(out)
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health": return self._send(200, {"ok": True, "version": __version__})
        if self.path == "/status":
            return self._send(200, {k: STATE[k] for k in ("state", "log", "error")} |
                              {"events": len(STATE["events"]), "threads_meta": len(STATE["meta"])})
        if self.path == "/result":
            return self._send(200, STATE["result"]) if STATE["result"] else self._send(404, {"error": "no result"})
        if self.path == "/seed":
            return self._send(200, STATE["seed_out"], "application/octet-stream") if STATE["seed_out"] else self._send(404, {"error": "no seed"})
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        if self.path == "/seed":
            b = self._body(); STATE["seed_in"] = b or None
            return self._send(200, {"seed_bytes": len(b)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/reset":
            with LOCK:
                if STATE["state"] == "running": return self._send(409, {"error": "running"})
                STATE.update(events=[], meta={}, seed_in=None, seed_out=None, state="idle", log=[], error=None, result=None)
            return self._send(200, {"ok": True})
        if self.path == "/ingest":
            d = json.loads(self._body() or b"{}")
            for e in d.get("events", []):
                STATE["events"].append((e[0], e[1] or "", e[2] or "", e[3] or ""))
            for t in d.get("threads", []):
                STATE["meta"][t[0]] = {"title": t[1], "account": t[2], "updated": t[3] or ""}
            return self._send(200, {"events": len(STATE["events"]), "threads": len(STATE["meta"])})
        if self.path == "/run":
            p = json.loads(self._body() or b"{}")
            with LOCK:
                if STATE["state"] == "running": return self._send(409, {"error": "already running"})
                STATE.update(state="running", error=None, result=None, seed_out=None)
            log("run started")
            threading.Thread(target=run, args=(p,), daemon=True).start()
            return self._send(202, {"state": "running"})
        self._send(404, {"error": "not found"})


def main():
    port = int(os.environ.get("PORT", "8080"))
    print(f"organism {__version__} listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
