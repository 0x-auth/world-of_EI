"""
genesis.py — a minimal "born with memory, learns by itself" intelligence.

Core idea (from BAZINGA-INTELLIGENCE-ROADMAP.md):
Standard AI: train on a huge corpus first -> then it's "born" and responds.
This system: born empty -> experiences one thing at a time -> remembers the
DELTA (change), not the raw input -> reconstructs an account of the past
when queried, instead of doing lookup/retrieval -> its own behavior shifts
based on which deltas repeat, with no separate "training phase" ever.

Deliberately NOT included (per BAZINGA corpus review): golden-ratio harmonics,
432Hz constants, DODO 5-dimension framework. Those are decorative in the
existing corpus, not load-bearing, and were explicitly excluded to keep this
simple and honest about what it actually does.
"""

from dataclasses import dataclass, field
from typing import Any
from collections import Counter
import json
import time


@dataclass
class Delta:
    """A single unit of memory: not the raw event, but what CHANGED because of it."""
    timestamp: float
    before: Any
    after: Any
    change: str  # human-readable description of what shifted

    def to_dict(self):
        return {"t": self.timestamp, "before": self.before,
                "after": self.after, "change": self.change}


class Genesis:
    """
    Born empty. No training data, no pretrained weights.
    Everything it "knows" comes from deltas it has personally experienced.
    """

    def __init__(self):
        self.state: dict = {}          # current understanding of the world
        self.memory: list[Delta] = []  # sequence of deltas, the ONLY long-term store
        self.pattern_counts: Counter = Counter()  # which changes repeat (this IS the learning)
        self.born_at = time.time()

    def experience(self, key: str, new_value: Any):
        """
        The only way information enters the system.
        Not 'training' — a single lived event, processed immediately.
        """
        old_value = self.state.get(key)
        change_desc = f"{key}: {old_value!r} -> {new_value!r}"

        delta = Delta(
            timestamp=time.time(),
            before=old_value,
            after=new_value,
            change=change_desc,
        )
        self.memory.append(delta)
        self.state[key] = new_value

        # Learning IS pattern frequency over lived deltas — no separate training step.
        self.pattern_counts[change_desc.split(":")[0]] += 1

    def reconstruct(self, key: str) -> str:
        """
        Answer 'what do you know about X' by RECONSTRUCTING an account from
        memory, not by looking up a stored fact. This is the roadmap's core
        distinction: memory = reconstruction, not retrieval.
        """
        relevant = [d for d in self.memory if d.change.startswith(f"{key}:")]
        if not relevant:
            return f"I have no experience of '{key}'."

        story = [f"I have experienced '{key}' change {len(relevant)} time(s):"]
        for d in relevant:
            story.append(f"  - {d.change}")
        story.append(f"Current understanding: {key} = {self.state.get(key)!r}")
        return "\n".join(story)

    def most_familiar(self, n: int = 3) -> list[tuple[str, int]]:
        """What has this system experienced most? This is its 'personality' —
        entirely a function of lived history, nothing pretrained."""
        return self.pattern_counts.most_common(n)

    def save(self, path: str):
        """Persist memory to disk — the thing the 2024 prototype's own
        TEST_RESULTS.md flagged as broken (state lost on shutdown)."""
        with open(path, "w") as f:
            json.dump({
                "born_at": self.born_at,
                "state": self.state,
                "memory": [d.to_dict() for d in self.memory],
                "pattern_counts": dict(self.pattern_counts),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Genesis":
        """Load a previous life. The system picks up exactly where it left off —
        no retraining, no re-initialization. This is the actual falsifiable
        claim: memory persists and continues shaping behavior across sessions."""
        g = cls()
        with open(path) as f:
            data = json.load(f)
        g.born_at = data["born_at"]
        g.state = data["state"]
        g.pattern_counts = Counter(data["pattern_counts"])
        g.memory = [Delta(d["t"], d["before"], d["after"], d["change"])
                    for d in data["memory"]]
        return g


if __name__ == "__main__":
    # Demo: born empty, given a handful of lived experiences, then queried.
    g = Genesis()

    g.experience("mood", "curious")
    g.experience("mood", "focused")
    g.experience("project", "genesis-intelligence")
    g.experience("mood", "focused")
    g.experience("location", "Pune")
    g.experience("mood", "focused")

    print(g.reconstruct("mood"))
    print()
    print("Most familiar patterns:", g.most_familiar())

    g.save("genesis_state.json")
    print("\nSaved. Reloading as a fresh instance to prove memory persists...")

    g2 = Genesis.load("genesis_state.json")
    print(g2.reconstruct("mood"))
