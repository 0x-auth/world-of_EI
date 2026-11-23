# The organism on Cloudflare

It lives next to its data. Nothing leaves your account; the container has no internet.

```
cron (hourly) ─► Durable Object "organism" ─► container (standard-2: 1 vCPU, 6 GiB)
                   │ reads D1 in pages             │ learns, links parts, keeps no data after
                   │ seed from R2 ───────────────► │
                   ◄─ links ─────────────────────── │
                   ▼                                 ▼
             D1: organism_runs, organism_links   R2 ei-organism: seed/latest.npz, seed/<run>.npz
```

**When it lives:** within the hour after a request, or once a day when the database has grown.
The first life starts within the hour after deploy (a newborn, no seed).

**What it reads (read-only):** `messages` (Claude, ChatGPT, Gemini, Gmail, WhatsApp, Telegram),
`github_commits` (one thread per repo), `authored_content` (one thread per piece). Long threads are cut into
sessions at 6-hour gaps, and links join 20-message parts.

**What it writes:** four new tables only: `organism_runs`, `organism_links`, `organism_strikes`,
`organism_control`. The Worker answers `/health` and nothing else, so no link is ever served publicly.

## Deploy (on your Mac, Docker or Colima running, wrangler logged in)

```bash
cd ~/github-repos/world-of_EI/organism/cloudflare
npm install
npx wrangler deploy
```

## Use

```sql
-- ask for a run now (picked up within the hour)
INSERT OR REPLACE INTO organism_control VALUES ('request', 'manual', datetime('now'));
-- the last run's links
SELECT rank, title_a, msgs_a, title_b, msgs_b, carried_by FROM organism_links
 WHERE run_id = (SELECT MAX(run_id) FROM organism_runs) ORDER BY rank LIMIT 50;
-- strike a wrong term pair: it never carries a link again, and is cut from the seed
INSERT OR REPLACE INTO organism_strikes VALUES ('xml', 'cookies', NULL, NULL, 'generic', datetime('now'));
-- if a life failed
SELECT * FROM organism_control;
```

## Cost

Workers Paid ($5/month) includes 25 GiB-hours of memory and 375 vCPU-minutes a month. One life on the full
database takes about 10 minutes on standard-2 (tested at 326k synthetic messages: 4.75 min on 2 cores, 2 GB peak).
Daily lives come to roughly 30 GiB-hours and 300 vCPU-minutes: well under a dollar a month above the plan.
