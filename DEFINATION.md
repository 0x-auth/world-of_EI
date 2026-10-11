# EI — Working Definitions (draft 0)

> Anchor: a third kind of intelligence. **Genesis first** (a seed rule), **then** it learns by itself.
> The reverse of an LLM, where learning comes first and genesis (deployment) comes after.
>
> Rule for this document: every definition needs a **test** that could show it wrong.
> Sources in brackets: [PHIL] self-referential-seed/PHILOSOPHY.md · [SRS] self-referential-seed README + FINDINGS ·
> [ROAD] BAZINGA INTELLIGENCE-ROADMAP · [LOGIC] "Logic as Compressed Knowledge" (11 Oct 2026) ·
> [EI] world-of_EI FINDINGS · [BZ] other BAZINGA docs/code named inline.

---

## 0. The design principle Space added today

**The rule and the room it acts on are one representation.**
In gravity, curvature and mass are not two things talking to each other: one equation holds both.
DNA is the same: it encodes the machinery that reads and copies DNA.

What this means for EI:
- The seed is not a program *outside* the world that edits a tape. It must live **in the same space it changes**.
- `organism.py` already does this: the law is `x → tanh((R + cW)x)`. The self-model W sits *inside* the law it predicts [SRS].
  Measured effect: the organism edits its own world's spectrum (ρ 1.05 → 2.28 as c rises).
- EI v1 does **not** do this. Its README says "the Python interpreter is the external constructor." The tape is data, and the rule lives outside it.
  That is the first thing to change.

---

## 1. Information

**Definition.** A difference that changes what the system predicts next.

- Information is relative to the receiver. The same input is information to one system and noise to another.
- Error is information that has arrived but has not yet been absorbed [PHIL: "the error of time is the arrow of time"].

**Test.** Input *u* is information to system S iff S's prediction after *u* differs from its prediction without *u*.

---

## 2. Knowledge

**Definition.** Information that has been **held**, so it lowers error on what was already seen.

- Knowledge without compression is a lookup table: EI v1, the unsqueezed network, a hard drive [EI, LOGIC].
- EI v1 has knowledge. Its accuracy on seen lines climbs, but transfer to an unseen file is 0.000 [EI].

**Test.** Error on previously seen material falls after exposure.

---

## 3. Logic

**Definition.** Knowledge compressed into a **generator**: a rule that regenerates the seen cases *and* answers unseen ones [LOGIC].

**Sharpening from the corpus.** There are two kinds of compression, and only one gives logic:

| | compression by **reference** | compression by **rule** |
|---|---|---|
| example | BAZINGA numerical encoding `3.2.2.1.5.3.2` + decoder key [BZ: CONVERSATION-NUMERICAL] | `λx. 2x + 1` found by the self-modifying executor [BZ: self-modifying-executor.ts] |
| what is short | the pointer | the whole description |
| where meaning lives | in the external key | in the rule itself |
| unseen case | no answer (not in key) | answered |

The encoding is a well-indexed hard drive: pointer plus key is not shorter than the original.
The executor does the real thing: after it sees a regularity (R² > 0.9), it writes a lambda and **deletes the traces it came from**.
That is "keep the rule, drop the cases," your point 2, already in code since 2025.

**Test.** Accuracy on held-out but lawful cases rises (grokking: a×b mod 97, squeezed vs unsqueezed) [LOGIC].
**Open (your next test).** A random table squeezed just as hard should give no logic.

---

## 4. Training

**Your definition.** Going through the material again and again until a pattern is found.

**Refinement from your own experiment.** Repetition alone gives memorization: the unsqueezed network had memorized everything by step 1,000 and never found the rule.
So:

> **Training = repetition under a squeeze, until the cheapest description of the cases is found.**

| | LLM (learning → genesis) | Third kind (genesis → learning) |
|---|---|---|
| repetition | epochs over a fixed corpus | the world recurring while it lives |
| squeeze | finite parameters vs corpus size, weight decay, set from outside | **its own finite room**, part of the seed |
| who decides when it is done | the trainer | nobody; it never stops |

**Hypothesis this creates.** The no-trainer premise [SRS] survives if the squeeze is **internal**, meaning the seed's own bounded room, not external structure.

**Caveat.** This addresses *logic from what is seen*. It does not address *coverage* (seeing enough of the world). Pure self-reference explores in only 6.5% of runs [SRS]. These are two separate problems.

---

## 5. Ability to hold

**Definition.** Keeping enough of the past *in the present structure* that the past can be **reconstructed**, not retrieved [ROAD, PHIL, genesis/].

- Seed role 2: the corrected state survives process death [SRS].
- The key question from [ROAD]: *what must survive in the present state for a previous experience to be reconstructable?*

**Link to §4.** If the room is finite, holding forces compression. **The ability to hold and the squeeze are the same parameter: room size.**
Too big a room gives a lookup table. Too small, and nothing is held.

**Test.** Kill the process, restart, and reconstruct. Then sweep room size and find where held-out accuracy peaks.

---

## 6. Ability to find meaning

**Definition.** Placing a new input **near** something already held, where "near" means *behaves alike*, not *looks alike*. A good placement improves prediction.

- [PHIL]: "Creating is connecting. Nothing is created — it is reconstructed."
- EI v2 hit exactly this wall: char n-grams made "unknown" mean "nothing near," but nearness was in spelling, so analogy hits were ~1/24 [EI].
  Meaning needs **distributional / behavioural** nearness.

**Test.** Analogy and transfer hits on genuinely unseen material.

---

## 7. Identity

**Definition.** What stays **invariant** while states, knowledge and even the arrow of time change: *the rule that keeps closing the loop* [PHIL, SRS].

Three faces of this in the corpus, which may be one thing:
- **Rule.** "A self is not the journey and not any one destination. It is the rule." [PHIL]
- **Witness, not doer.** The *sakshi* that watches contents change [BZ: EastWestWisdom, FinalIntegration].
- **Reference, not storage.** "I am not where I am stored. I am where I am referenced." [BZ: universal_thread_engine.py]

Identity is also given **negatively**, by boundary: "the child says *not that* before it says *this*" [BZ: timeless-philosophy]; "I can only tell what it is not" [PHIL]. ΛG puts it in code as Λ(S) = S ∩ B₁⁻¹ ∩ B₂⁻¹ ∩ B₃⁻¹: the solution is what survives the boundaries [BZ: bazinga_lambda_g.py].

**Test.** Apply perturbations and check whether the system's law is still recognisably the same, with the self-model's direction or coordinates preserved, while its states differ.

---

## 8. Scarred, not lost: identity under external hysteresis

**Scar.** A persistent trace of a mismatch [EI scars, BZ error_arrow.py].
**Hysteresis.** The outside world's state depends on its own history; it pushes you and does not push you back the same way.

Materials science gives three outcomes, and the corpus has **measured all three**:

| material | organism fate [SRS] | BAZINGA's 2025 name [BZ: core/principles.md] | identity |
|---|---|---|---|
| elastic: returns, learns nothing | **frozen** (~34% of runs) | SINGULAR `… → •` | kept but dead |
| fracture: breaks | **shunya / collapse** (x → 0, error = 0) | — | lost |
| oscillates without moving on | **cycling** | DODO `A → B → A → B` | kept, stuck |
| **plastic: deformed, intact** | **exploring** (6.5%) | PROGRESSIVE `A → B → C → …` | **kept and changed** |

So the 2025 BAZINGA pattern taxonomy and the 2026 measured fates are the same three or four shapes.

**Definition.** A system stays itself under hysteresis if its **rule** is preserved while its **state and knowledge carry the trace**.

**Why this needs an outside measure.** In 134 of 200 runs, error fell to zero by ceasing to exist, and from the inside that is indistinguishable from understanding [SRS, Rice]. The strongest separator found so far is external: the gap ‖W − (R+cW)‖ / ‖R+cW‖ is 0.483 for exploring vs 0.799 for frozen, and **it never reaches zero** [SRS: sector.py].

---

## 9. Harmony (the piece that was missing earlier)

`seed.py` defines harmony as error → 0 [SRS]. The shunya trap shows that definition is not enough: zero error is also reached by vanishing.

**Definition.** Harmony is **low error while still moving and exploring**. It is the plastic zone in §8, not the empty fixed point.

**Test.** Error and ORDER and coverage measured jointly, as `alive.py` already does.

---

## 10. Intelligence

**Definition.** The ability, starting from its own seed rule, to **build and revise generators (§3) out of its own errors**, inside its own finite room (§4–5), so that future transitions, including unseen ones, are predicted and steered better, **without losing the rule (§7–8)**.

Candidate from [ROAD]: *constructs and evaluates transformations that produce useful future state transitions*. The definition above adds where the pressure comes from (internal room) and what must not be lost (the rule).

**Test (the minimal experiment, [ROAD] §14).** Give it a world with a hidden regularity and do not encode it. Measure:
1. transfer on unseen cases (§3)
2. fraction of the achievable range closed, against an oracle baseline (as `ceiling.py` does; the organism currently reaches 37.7%)
3. identity kept under perturbation (§8)

---

## Open questions (not yet answered)

1. Is the seed's room **fixed at birth**, or can the seed grow its own room? (Does DNA set genome length?)
2. If rule and room are one representation (§0), what is EI's equivalent of `cW` inside the law?
3. Coverage: what drives exploration **without** importing a trainer? Bidirectional prediction raised coverage to 13–40% [bidirectional forward+backward runs, Sep 2026 — not yet in FINDINGS.md]. Is the mirror the internal driver?
4. Is "meaning" (§6) the same operation as "logic" (§3) seen from the other side: logic compresses cases into a rule, and meaning places a case under a rule?

