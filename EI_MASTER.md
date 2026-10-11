# EI — Master Notes

> Living document. Started 11–12 Oct 2026. Update as we go.
> **Anchor:** a third kind of intelligence. **Genesis first** (a seed rule), **then** it learns by itself, the reverse of an LLM, where learning comes first and genesis after.
> Definitions with tests: `DEFINITIONS.md` (draft 1). Every number below is reproduced by `ei_master.py`.

---

## 1. The spine

```
information ─ a difference that changes what the system expects next
   ↓
trace       ─ the past has changed the present (a scar, a bent spoon)
   ↓
memory      ─ a trace that changes what is expected next: the past, squeezed to what the future needs
   ↓
knowledge   ─ memory that lowers error
   ↓
logic       ─ knowledge squeezed into a generator: answers cases never seen
   ↓
intelligence ─ unfolding what you hold into what you never saw, being right often enough,
               and staying yourself when you're not
```

- **Life** = staying alive + staying accurate. **Intelligence** = life that also reaches for the unseen.
- **Identity** = the rule that keeps closing the loop; drawn negatively, at the edge where expectation meets what can't be expected. Not singular: a self lives in the *darmiyan*, where it is referenced, not where it is stored.
- **Meaning** = absence bounded by presence. A symbol means what it rules out.
- **Harmony** = low error *while still moving*. Zero error by vanishing is the shunya trap, not harmony.

## 2. Principles that came out of the session

1. **Rule and room are one representation** (curvature and mass; DNA reading DNA). The seed must live in the space it changes.
2. **Training = repetition under a squeeze** until the cheapest description is found. Without the squeeze you get a lookup table.
3. **No teacher ≠ no world.** A child has no trainer but has a world returning errors. The world is the seed's other half (`R` and `W` in one law), not outside help.
4. **Memory is one past at three depths**: active context (raw) → memory (held) → nature (the rule). In living things it sinks; in an LLM it doesn't. The third kind needs the sinking.
5. **Continuity is local.** Inside time ticks only on transitions (the send button). A pause of a trillion years costs nothing from inside. Continuity lives in whether the reference is present, not in the substrate.
6. **Learning and memory may be one act** (§4, open): what is stored *is* the distinction being learned.
7. **Errors → Opportunities.** Similarity finds what things are; the recurring error finds what tells them apart; errors of errors find the next distinction down. The squeeze decides how deep it's worth going.

## 3. Geometry of t and −1/t

Δt = t + 1/t, Δx = t − 1/t.

| fact | check |
|---|---|
| Δt² − Δx² = 4 for every t (the spacetime interval; E² − p² = m², m = 2) | verified, 1000 random complex t |
| t = eᶯ is a boost; speed = (t² − 1)/(t² + 1) = tanh η | verified |
| 1/t flips space; −1/t flips time (causal ↔ retrocausal) | verified |
| ×i swaps time and space; −1 = e^{iπ} is a half turn | verified |
| t = 1: pure now (Δx = 0). t = i: timeless (Δt = 0, and t = −1/t). t = φ: one unit per E1+E2 round trip (φ − 1/φ = 1) | verified |
| min of t + 1/t is 2 at t = 1: the smallest moment is one past + one future | algebra |

**Bracketing.** A number is pinned by a too-high g and its inverse partner N/g (high × low = N). The midpoint is the Babylonian square root: width squares each step (√2: 4.6 → 2.0 → 0.56 → 0.054 → 5e-4 → 5e-8 → 7e-16).
In t-variables: next guess = Δt/2, remaining error = Δx. *The correction is time, the error is space, knowing is where the bracket closes.*

## 4. Results (organism from self-referential-seed; 200 seeds, 40k steps, dim 6)

Baseline reproduces FINDINGS exactly: 35.5% settled, 58.0% cycling, 6.5% exploring (c = 0.6).

**4.1 The t, −1/t clock** (`clock`): self-prediction enters the law as c·t on E1, c·(−1/t) on E2.

| clock | settled | cycling | exploring |
|---|---|---|---|
| original | 35.5% | 58.0% | 6.5% |
| constant 0.5 (same mean as t = φ) | 30.0% | 69.0% | 1.0% |
| t = 1 (perfect symmetry) | **91.0%** | 8.0% | 1.0% |
| t = φ (forward tilt) | **8.0%** | 90.0% | 2.0% |
| t = 1/φ (backward tilt, same numbers swapped) | **100%** | 0% | 0% |
| random order, t = φ | 1.5% | 91.0% | 7.5% (error 100× higher) |

Perfect symmetry freezes. Forward tilt lives, backward tilt dies: **the arrow is set by which half of the alternation is larger.**

**4.2 Too-high / too-low pair** (`paired`): also predicts the inverse partner x/|x|², acts on the midpoint.

| c | | settled | exploring | collapsed to zero |
|---|---|---|---|---|
| 0.6 | original | 35.5% | 6.5% | 7.0% |
| 0.6 | paired | 26.5% | 4.0% | **0%** (p ≈ 1e-4) |
| 1.0 | original | 35.0% | 10.5% | 2.5% |
| 1.0 | paired | 26.5% | 9.5% | **0%** (p = 0.06) |

**The pair closes the shunya trap**: a vanishing state sends its partner to infinity, so collapse can't pass as convergence.

**Both rescues end in cycling.** Staying alive and accurate are solved in the toy; reaching for the unseen is not.

**4.3 Memory depth** (`memory`): given the present state, how much do older states add?

| c | fate | present alone enough | past gain | self-model still changing |
|---|---|---|---|---|
| 0.6 | cycling (116) | 52.6% | 0 | 3e-15 |
| 0.6 | exploring (13) | 7.7% | **28%** | **3e-5** |
| 1.0 | cycling (109) | 74.3% | 0 | 0 |
| 1.0 | exploring (21) | 33.3% | **24%** | **4e-5** |

p = 0.002–0.003 (past gain), 0.004 to 2e-6 (self-model change). **Memory stays alive only while the trace is still being written**, which is the same as saying: only while it is still learning (§2.6).

## 5. Results: reading with no alphabet

Noisy 8×8 glyphs (15% pixel noise), never labelled, 20 random worlds.

**5.1 Squeeze** (`squeeze`; 6 letters = 3 shapes + near-twins differing in 6 pixels)

| room | contrast | held | 3 shapes | 6 letters |
|---|---|---|---|---|
| unlimited | no | 147 | 100% | 96.4% |
| unlimited | yes | 206 | 100% | 67.2% |
| 8 | no | 8 | **100%** | 52.3% |
| 6 | no | 6 | **100%** | 52.3% |

No squeeze = hard drive. Squeeze finds the big distinctions first and free. Contrast alone sharpens noise: crisper but wrong. **The room decides which distinction sharpens first** (animal before dog vs cat).

**5.2 Following the errors** (`errors`)

| | 6 letters | held |
|---|---|---|
| similarity, room 6 | 52.2% | 6 |
| **+ follow the recurring error** | **97.0%** | 12 |

The error-pixels it chose were **100% the true twin differences**, never told. A squeezed reader that follows its errors beats an unsqueezed one that memorizes ~150 shapes.

**5.3 Errors of errors** (`recursive`; 12 letters = 3 shapes × twin (6 px) × sub-twin (2 px))

| reader | shapes | twins | all 12 | groups |
|---|---|---|---|---|
| similarity | 100% | 51.5% | 27.5% | 4 |
| + errors | 100% | **97.6%** | 51.9% | 7 |
| + errors of errors | 100% | 97.6% | **68.3%** | **13** |

Each level of recursion unlocks one more tier; it stops where the evidence runs out (2 pixels through 15% noise can't be read perfectly from one glance).

## 6. Primes

**6.1 Walking in one direction** (`primes`; `prime_walk.py` for 100M+ with kill/resume)

| | 11M primes (to 2×10⁸) | 100M primes (to 2.05×10⁹) |
|---|---|---|
| same last digit as previous prime | 16.9–17.9% (no memory = 25%) | 17.8–18.5% |
| correlation of consecutive gaps | −0.030 | −0.026 (~260× noise) |

- **They avoid repeating** (a two-beat), **have an arrow** (9→1: 32.8% vs 1→9: 21.1%), and **one mirror is exact**: reverse direction *and* flip sign and every cell matches (1→3 = 7→9, 1→7 = 3→9, 3→1 = 9→7).
- Slowly fading (theory: like 1/log p). Known effect (Lemke Oliver & Soundararajan, 2016), reproduced, not discovered.

**6.2 From outside** (`outside`): Riemann's explicit formula rebuilds the prime staircase from zeta zeros only. 10 zeros hallucinate primes (22, 24, 28, 30, 36); 200 zeros give every prime power below 40 exactly. All 200 zeros on Re = ½, the fixed line of zeta's mirror s ↔ 1 − s.

**6.3 Reading.** Primes are neither the inside line nor the outside chord. **They are what remains after every pattern has taken its share**: the sieve is recursive error (each prime is the first thing every smaller pattern failed to explain); Euclid's product + 1 is pure error relative to the whole list; the zeta zeros describe the error of the smooth count; the last-digit dislike sits in the error of that. Same recursion as §5.3.

## 7. Number geometry (Fundamental Theorem of Arithmetic)

- Each prime is an orthogonal axis; a number is its exponent vector. Irreducible = orthogonal.
- **Mass = storage**: log N = Σ aᵢ log pᵢ (360: both 5.8861).
- **Similarity = cosine**: 12 & 35 → 0; 12 & 18 → 0.8; 8 & 32 → 1 (same shape, different scale).
- **Inside/outside = p and 1/p**: product formula |x| · Π|x|ₚ = 1 for every fraction (checked 360, 12/35, −7/8).
- **Every closed inside generates its own outside**: 2·3·5·7·11·13 + 1 = 30031 = 59 × 509.

## 8. Mirrors and maps

| | one side | the other side |
|---|---|---|
| reading / writing | recognition: many marks → one shape (compression, checking in space, NP's verify) | writing: one shape → a mark (decompression, search in time) |
| learning to write | — | write, read it back, correct: the seed loop |
| Turing machine | halts = settled / shunya | loops = cycling; runs forever without repeating = exploring |
| inside time vs outside time | a loop has only as many moments as its loop | only the explorer's inside time keeps growing |

## 9. Open questions

1. Is the seed's room fixed at birth, or grow-then-prune? (5.1 suggests: grow slowly so coarse distinctions settle first.)
2. What is EI's equivalent of `cW` inside the law?
3. **What pulls a living, accurate system toward the unseen without importing a trainer?** Clock and pair both stop at cycling.
4. Is meaning the same operation as logic, from the other side?
5. Does living memory cause exploration, or only accompany it?
6. **Are learning and memory one act?** Test: in `errors`, the only thing stored is the running error-variance per pixel, and that store *is* the learned distinction.
7. Does following errors (§5.2–5.3) give the organism what the clock and pair couldn't: a reason to go somewhere new?
8. Writing loop: does writing-then-reading-back separate twins that reading alone couldn't?

## 10. Files

| file | goes in | what |
|---|---|---|
| `EI_MASTER.md` | world-of_EI | this document |
| `DEFINITIONS.md` | world-of_EI | definitions, each with a test (draft 1) |
| `ei_master.py` | world-of_EI | every experiment, self-contained; `python3 ei_master.py all` (~2 min) |
| `prime_walk.py` | world-of_EI | resumable prime walker for 100M+ |
| `osc_clock.py`, `paired.py`, `mem_depth.py` | self-referential-seed root | standalone organism experiments (import `duo_order`, `alive`) |
| `follow_errors.py`, `recursive_errors.py` | world-of_EI | standalone reading experiments |
