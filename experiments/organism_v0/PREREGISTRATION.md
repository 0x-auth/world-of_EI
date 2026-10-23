# Pre-registration — EI organism v0 on Space's digital history

Written 2026-10-19, **before** any model has seen the test window. Hyperparameters are tuned only on the validation month; the test is run once.

## Question
An organism is born with an empty map plus one seed rule. It lives through Space's own words in time order and learns only from its own prediction errors. Does its map predict **unseen future months** better than a static snapshot built from the same past?

## Data (local export of D1 `digital-history`, read-only)
- **Stream:** every message Space wrote, in one time-ordered stream:
  - to Claude (source 4, role `human`)
  - to ChatGPT (source 5, roles `user`/`human`)
  - his commit messages (`github_commits`)
- **Cleaning:**
  - rows without a timestamp are dropped
  - exact duplicate texts of 40+ characters are dropped after their first occurrence (this stops leakage across imported accounts)
  - only the first 2000 characters of each message are used
- **Windows:**
  - life: 2024-06-01 → 2025-04-30 (the organism lives and learns)
  - validation: 2025-05 (tuning only)
  - **test: 2025-06-01 → 2025-08-31** (run once)
- **Terms:**
  - lowercased words matching `[a-z][a-z0-9_'-]{2,}`, plus the symbols φ π Δ ∆ ∞ ψ Ω τ θ
  - English stopwords removed
  - vocabulary = the 4000 terms with the highest message frequency in **life only**, each appearing in at least 5 messages

## Task
Before reading message t+1, predict which terms it will contain. Every model outputs a distribution p over the vocabulary.

**Score:** mean over the target's unique in-vocabulary terms of −log₂ p(w), averaged over messages (bits; lower = less surprised). Secondary score: recall@20. Targets with no in-vocabulary term are skipped (the same set for all models).

## Context
Every model that conditions on context reads the same residue trace, c ← α·c + (1−α)·x_t, L1-normalised. Here x_t is the current message's term indicator.

## Models
| name | what it is | learns during test? |
|---|---|---|
| U-frozen | message frequencies of terms over life | no |
| S-frozen (**the snapshot**) | λ·(c-weighted next-message co-occurrence counts over life) + (1−λ)·U | no |
| O-frozen (**the organism**) | softmax(b + Wᵀc). Born with W = s·I (seed rule: "what I just said, I'll say again") and b = 0. Error-driven online updates during life: ΔW = η·c(y−p)ᵀ, Δb = η(y−p), where y is the normalised next-message terms | no (weights frozen at 2025-05-31) |
| U-alive, S-alive | the same counts, still accumulating through test (predict first, then count) | yes |
| O-alive | the organism keeps living (predict first, then learn) | yes |

## Hypotheses
- **H1** (map beats snapshot): bits(O-frozen) < bits(S-frozen) on test.
- **H2** (being alive matters beyond counting): bits(O-alive) < bits(S-alive) on test.
- **Kill condition:** if O-frozen is not below S-frozen, the error-driven map adds nothing over a static co-occurrence snapshot at v0.

## Uncertainty
Block bootstrap over test days (1000 resamples) of the per-message bit difference; 95% CI reported.

## Tuning grid (validation month only)
η ∈ {2, 5, 10, 20, 40}, α ∈ {0, 0.5, 0.8}, s ∈ {0, 2} for the organism; λ ∈ {0.3, 0.5, 0.7, 0.9} for the snapshot, with the same α grid.

## Amendment 1 (2026-10-19, before any test-window run)
On validation the organism improves monotonically as η falls, and the optimum sits at the grid edge (η = 2). The grid is extended downward to η ∈ {0.25, 0.5, 1}, with the same α and s, still scored on **validation only**. If the snapshot's best λ also lands on an edge, its grid is extended the same way. The test window is still unseen and is run once after this.
