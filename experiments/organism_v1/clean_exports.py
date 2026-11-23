#!/usr/bin/env python3
"""Clean Claude exports into plain chats the organism can read.

Usage:
  python3 ~/Downloads/clean_exports.py ~/Downloads/06-11-2026-claude-exports ~/Downloads/clean

Input : one folder per account, each with conversations.json (raw export).
Output: <out>/<account>/conversations.json  — same top shape, but every message is just
        {uuid, sender, created_at, text}; artifacts are extra messages with sender "artifact".
        <out>/artifacts/<account>/<chat>/<file>  — every artifact rebuilt as a file.
        <out>/clean_report.txt — what was kept, dropped, rebuilt, redacted.

What it does:
  * follows only the branch you ended on (parent_message_uuid), drops retried/edited branches
  * keeps your words, Claude's text blocks, voice notes, attachment text
  * drops thinking, tool results (search pages, bash output), token_budget, flags, injected prompts,
    and the "This block is not supported on your current device yet." placeholder
  * rebuilds artifacts by replaying file writes and patches in order:
      create_file(path, file_text), Write(file_path, content), artifacts(create/rewrite/update),
      str_replace(path, old_str, new_str), Edit(file_path, old_string, new_string)
    so files that were only "presented" get their text back
  * strips <style>/<script> and tags from HTML for the text the organism reads (files on disk are kept whole)
  * redacts secrets (GitHub/Slack/OpenAI/Anthropic/Google keys, bot tokens, bearer tokens, long hex/base64 keys)
  * leaves out chats whose title matches --exclude (default: the chat that built this)
"""
import sys, os, re, json, argparse, collections

PLACEHOLDER = re.compile(r"`{3}\s*This block is not supported on your current device yet\.\s*`{3}|"
                         r"This block is not supported on your current device yet\.")
SECRETS = [re.compile(p) for p in [
    r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{40,}", r"xox[abpsr]-[A-Za-z0-9-]{10,}",
    r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}", r"AIza[0-9A-Za-z_-]{35}", r"ya29\.[0-9A-Za-z_-]{20,}",
    r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b", r"(?i)bearer\s+(?!\{\{)[A-Za-z0-9._~+/=-]{20,}",
    r"(?i)(?:api[_-]?key|secret|token|password)[\"']?\s*[:=]\s*[\"'](?!\{\{)[^\"'\s]{12,}[\"']",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"]]
EXCLUDE_DEFAULT = "Cross-account chat synthesis"
stats = collections.Counter()


def redact(t):
    for p in SECRETS:
        t, n = p.subn("[REDACTED]", t); stats["secrets redacted"] += n
    return t


def strip_markup(t):
    t = re.sub(r"(?is)<(style|script)[^>]*>.*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"[ \t]+", " ", t)


def active_branch(msgs):
    if not msgs: return []
    by = {m["uuid"]: m for m in msgs}
    cur, path = max(msgs, key=lambda m: m.get("created_at", "")), []
    while cur is not None and len(path) <= len(msgs):
        path.append(cur); cur = by.get(cur.get("parent_message_uuid"))
    stats["messages on dropped branches"] += len(msgs) - len(path)
    return path[::-1]


def apply_tool(b, files):
    """Replay one tool call onto the chat's file store. Returns the path touched, or None."""
    name, inp = b.get("name"), b.get("input") or {}
    if not isinstance(inp, dict): return None
    if name == "create_file" and isinstance(inp.get("file_text"), str):
        files[inp.get("path") or "file"] = inp["file_text"]; return inp.get("path")
    if name == "Write" and isinstance(inp.get("content"), str):
        files[inp.get("file_path") or "file"] = inp["content"]; return inp.get("file_path")
    if name == "artifacts":
        key = inp.get("title") or inp.get("id") or "artifact"
        key = files.get("__ids__", {}).get(inp.get("id"), key)
        files.setdefault("__ids__", {})[inp.get("id")] = key
        if inp.get("command") in ("create", "rewrite") and isinstance(inp.get("content"), str):
            files[key] = inp["content"]; return key
        if inp.get("command") == "update" and key in files and inp.get("old_str") is not None:
            files[key] = files[key].replace(inp["old_str"], inp.get("new_str", ""), 1); return key
    if name in ("str_replace", "Edit"):
        p = inp.get("path") or inp.get("file_path")
        old = inp.get("old_str", inp.get("old_string")); new = inp.get("new_str", inp.get("new_string", ""))
        if p in files and old is not None:
            if old in files[p]:
                files[p] = files[p].replace(old, new, -1 if inp.get("replace_all") else 1); return p
            stats["patches that did not apply"] += 1
    return None


def clean_message(m, files, touched):
    role = m.get("sender") or m.get("role")
    blocks = m.get("content") if isinstance(m.get("content"), list) else []
    parts = []
    for b in blocks:
        t = b.get("type")
        if t == "text":
            parts.append(b.get("text") or "")
        elif t == "voice_note" and role == "human":
            parts.append(b.get("text") or "")
        elif t == "tool_use":
            p = apply_tool(b, files)
            if p: touched[p] = m.get("created_at", "")
        elif t in ("thinking", "tool_result", "token_budget", "flag", "injected_prompt_block"):
            stats[f"dropped {t} blocks"] += 1
    if not blocks and m.get("text"):
        parts.append(m["text"])
    for a in m.get("attachments") or []:
        if role == "human" and a.get("extracted_content"):
            parts.append(f"[attachment {a.get('file_name', '')}]\n{a['extracted_content']}")
            stats["attachments kept"] += 1
    text = "\n".join(parts)
    text, n = PLACEHOLDER.subn("", text); stats["placeholders removed"] += n
    return redact(text).strip()


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:80] or "untitled"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--exclude", default=EXCLUDE_DEFAULT, help="comma-separated title fragments to leave out")
    a = ap.parse_args()
    src, out = os.path.expanduser(a.src), os.path.expanduser(a.out)
    excl = [x.strip().lower() for x in a.exclude.split(",") if x.strip()]
    report = []
    for acct in sorted(os.listdir(src)):
        p = os.path.join(src, acct, "conversations.json")
        if not os.path.exists(p): continue
        raw = json.load(open(p, encoding="utf-8"))
        before = stats.copy()
        clean = []
        for c in raw:
            title = c.get("name") or "Untitled"
            if any(x in title.lower() for x in excl):
                stats["chats excluded"] += 1; report.append(f"  excluded: {acct} · {title}"); continue
            files, touched, msgs = {}, {}, []
            for m in active_branch(c.get("chat_messages", [])):
                t = clean_message(m, files, touched)
                if t:
                    msgs.append({"uuid": m.get("uuid"), "sender": m.get("sender"),
                                 "created_at": m.get("created_at", ""), "text": t})
            files.pop("__ids__", None)
            adir = os.path.join(out, "artifacts", acct, safe(f"{(c.get('updated_at') or '')[:10]}_{title}"))
            for path, body in files.items():
                body = redact(body)
                os.makedirs(adir, exist_ok=True)
                with open(os.path.join(adir, safe(os.path.basename(path))), "w", encoding="utf-8") as f:
                    f.write(body)
                msgs.append({"uuid": None, "sender": "artifact", "created_at": touched.get(path, ""),
                             "text": f"{os.path.basename(path)}\n{strip_markup(body)}"})
                stats["artifacts rebuilt"] += 1
            msgs.sort(key=lambda x: x["created_at"])
            if msgs:
                clean.append({"uuid": c.get("uuid"), "name": title, "summary": c.get("summary", ""),
                              "created_at": c.get("created_at"), "updated_at": c.get("updated_at"),
                              "account": acct, "chat_messages": msgs})
            else:
                stats["empty chats dropped"] += 1
        os.makedirs(os.path.join(out, acct), exist_ok=True)
        with open(os.path.join(out, acct, "conversations.json"), "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False)
        d = stats - before
        report.append(f"{acct}: {len(raw)} chats in, {len(clean)} out · " +
                      " · ".join(f"{k} {v}" for k, v in sorted(d.items())))
    with open(os.path.join(out, "clean_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n\nTOTAL\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(stats.items())) + "\n")
    print("\n".join(report)); print(f"\nwrote {out}/<account>/conversations.json, {out}/artifacts/, {out}/clean_report.txt")


if __name__ == "__main__":
    main()
