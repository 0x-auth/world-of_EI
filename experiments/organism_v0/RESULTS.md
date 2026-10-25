# EI organism v0: results (test run once, 2026-10-19)

Protocol: `PREREGISTRATION.md` (plus Amendment 1, which extended the tuning grids before any test run).

**Stream:** Space's own words in time order, 48,928 messages:
- his messages to Claude and to ChatGPT
- his commit messages

**Windows:**
- life: Jun 2024 → Apr 2025
- validation: May 2025
- **blind test: Jun → Aug 2025** (16,806 messages)

**Vocabulary:** 4000 terms. A uniform guess scores 11.97 bits.

## Picked on validation
- organism: η = 1, α = 0, seed rule on (W₀ = 2·I)
- snapshot: α = 0.5, λ = 0.99

## Test (bits per term of the next message; lower = less surprised)
| model | bits | recall@20 |
|---|---|---|
| U-frozen (frequencies) | 10.476 | 0.141 |
| S-frozen (**snapshot**) | 10.319 | 0.160 |
| **O-frozen (organism, stopped learning 31 May)** | **10.096** | 0.157 |
| U-alive | 10.297 | 0.155 |
| S-alive (snapshot keeps counting) | 10.103 | 0.175 |
| **O-alive (organism keeps living)** | **9.720** | **0.206** |

Day-block bootstrap, 95% CI:
- **H1** O-frozen − S-frozen = **−0.223 bits [−0.242, −0.203]**. Holds.
- **H2** O-alive − S-alive = **−0.382 bits [−0.408, −0.356]**. Holds.
- O-alive − O-frozen = −0.376 [−0.405, −0.348]. Staying alive is worth as much again.

By month, the frozen organism's lead over the snapshot does not fade (Jun 0.225, Jul 0.223, Aug 0.222 bits). The alive organism keeps improving: 9.82 → 9.66 → 9.64.

Mixed point: on recall@20 the frozen organism is level with the snapshot (0.157 vs 0.160). Its win is in calibration (how much probability it puts on the right terms), not in the top-20 list.

## What the map holds (`map_reading.txt`)
- **The seed rule survived life:** the diagonal (self-persistence) is 2.08, against an off-diagonal mean of 0.00. The map is sparse: only 0.005 % of links exceed 0.5.
- **Strongest learned links** are mostly the rhythm of working with an AI:
  - artifacts → continue, thanks → continue, continue → finish, retry → pls
  - a few are content: ∞ ↔ π, time → space, question → answer, bazinga → friend
- **First spectral axis:** a *doing* pole (continue, file, update, bash, artifacts) against a *thinking* pole (like, know, think, mean, right, things). The map's main split is two modes of the same person.

## Limits of v0
- The terms are bags of words. Meaning beyond co-occurrence is not modelled.
- Commit messages include repos that may not be his alone (e.g. co-authored lines).
- One person's own stream only. WhatsApp (the only multi-person source) is the planned v1 stream.
