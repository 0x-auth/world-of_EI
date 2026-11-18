# Input format

## 1. Raw conversation (input to `ingest.py`)

A JSON array of message objects, in any order. Fields, with the alternatives `ingest.py` also accepts:

| field | required | alternatives | meaning |
|---|---|---|---|
| `ts` | yes | `time`, `date` | ISO time, e.g. `2024-05-04T13:20:00+05:30` |
| `text` | yes | `body`, `message` | the message; `null` or empty is kept as `⟨null⟩` |
| `direction` | one of these two | | `sent` / `received` (also `out`/`in`, `me`/`them`) |
| `sender` | one of these two | `from`, `who` | sender name or address, matched with `--me-pattern` |
| `source` | no | | `whatsapp`, `gmail`, … (anything starting with `wh` is chat; everything else is email) |
| `subject` | no | | email subject; kept for new threads, dropped for `Re:`, cut to one line for `Fwd:` |
| `message_id` | no | | used to remove duplicates (the same mail stored in two accounts) |

Minimal example:

```json
[
  {"ts": "2024-05-04T13:20:00+05:30", "direction": "sent",     "source": "whatsapp", "text": "lunch kiya?"},
  {"ts": "2024-05-04T13:25:00+05:30", "direction": "received", "source": "whatsapp", "text": "haan, dal chawal. you?"}
]
```

No `direction` field? Pass `--me-pattern "you@mail.com|Your Name"`, and every sender matching it becomes `me`.

What `ingest.py` does to the text:
- Email: removes quoted reply history (`On … wrote:`, lines starting with `>`) and signatures.
- Exact duplicates are dropped.
- Each message is cut to `--max` characters (default 1500).
- Marks are turned into tokens: `This message was deleted` → `⟨deleted⟩`, `image omitted` → `⟨media⟩`, attachments → `⟨file⟩`, links → `⟨url⟩`, `null` → `⟨null⟩`.

## 2. Timeline TSV (output of `ingest.py`, input to `convorg.py`)

One message per line, oldest first, tab-separated:

```
time              who    src  text
2024-05-04 13:20  me     w    lunch kiya?
2024-05-04 13:25  them   w    haan, dal chawal. you?
```

- `time`: `YYYY-MM-DD HH:MM`
- `who`: `me` or `them` (any two values work; give yours with `--me`)
- `src`: `w` for chat, `g` for email. It is only a tag: all sources form one timeline.
- `text`: no tabs or newlines

You can write this TSV yourself from any source and skip `ingest.py`. To add new messages, append them at the end and run `convorg.py learn` again.

## 3. Memory (`<conv>.organism.npz`)

| key | contents |
|---|---|
| `vocab` | the terms, in id order |
| `read` | how many TSV lines have been learned |
| `runs` | how many learning runs |
| `W_them`, `W_me`, `W_back` | term → term maps (float16) |
| `b_them`, `b_me`, `b_back` | biases |

Keep this file private. It is learned from the conversation.
