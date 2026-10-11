#!/usr/bin/env python3
"""
EI — Emergent Intelligence

A bounded self-referential learner.

The Python interpreter is the external constructor.
The JSON tape is persistent structural memory.
The generated program is a safe declarative program,
not arbitrary executable Python.

Learning:
    observe
    -> predict
    -> compare with reality
    -> write scar
    -> update world model
    -> synthesize safe program
    -> persist

Unknown is not automatically an error.
A genuine error occurs when an existing model predicts incorrectly.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9_'-]+")


class EI:
    def __init__(self, root: Path):
        self.root = root

        self.tape_path = root / "tape" / "self.json"
        self.program_path = root / "programs" / "transition_reasoner.json"
        self.results_path = root / "results" / "learning_curve.json"

        self.tape_path.parent.mkdir(parents=True, exist_ok=True)
        self.program_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

        self.tape = self._load_tape()

        # Active world model.
        # This is reconstructed from persistent structural memory.
        self.model: dict[str, Counter[str]] = defaultdict(Counter)

        for item in self.tape.get("knowledge", []):
            context = item.get("context")
            consequence = item.get("consequence")
            evidence = int(item.get("evidence", 1))

            if context and consequence:
                self.model[context][consequence] += max(evidence, 1)

        # Persistent scars are retained across restarts.
        self.scars: list[dict[str, Any]] = self.tape.get("scars", [])

        self.observations = 0
        self.unknowns = 0
        self.errors = 0
        self.correct = 0

    def _load_tape(self) -> dict[str, Any]:
        if not self.tape_path.exists():
            return self._new_tape()

        try:
            return json.loads(self.tape_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Warning: invalid tape; creating a new one.")
            return self._new_tape()

    @staticmethod
    def _new_tape() -> dict[str, Any]:
        return {
            "version": 1,
            "machine": "EI",
            "head": 0,
            "program": [],
            "knowledge": [],
            "scars": [],
            "metrics": {},
        }

    @staticmethod
    def lines(text: str) -> list[str]:
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    @staticmethod
    def tokens(line: str) -> list[str]:
        return [word.lower() for word in WORD_RE.findall(line)]

    def read_world(self, path: Path) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"World file not found: {path}")

        text = path.read_text(encoding="utf-8", errors="ignore")
        return self.lines(text)

    def contexts(self, line: str) -> list[tuple[str, str]]:
        """
        Extract one-word and two-word contexts.

        Example:
            "state becomes transition"

        Produces:
            ("state", "becomes")
            ("becomes", "transition")
            ("state becomes", "transition")
        """
        words = self.tokens(line)
        pairs: list[tuple[str, str]] = []

        for i in range(len(words) - 1):
            pairs.append((words[i], words[i + 1]))

        for i in range(len(words) - 2):
            context = f"{words[i]} {words[i + 1]}"
            pairs.append((context, words[i + 2]))

        return pairs

    def predict(self, context: str) -> str | None:
        choices = self.model.get(context)

        if not choices:
            return None

        return choices.most_common(1)[0][0]

    def observe(self, line: str, learn: bool = True) -> dict[str, Any]:
        """
        Observe one line.

        Unknown:
            no existing model for this context.

        Error:
            existing model predicted the wrong consequence.

        Scar:
            persistent record of a mismatch or first encounter.
        """
        line_unknowns = 0
        line_errors = 0
        line_correct = 0

        for context, actual in self.contexts(line):
            prediction = self.predict(context)
            self.observations += 1

            if prediction is None:
                # Unknown is uncertainty, not failure.
                line_unknowns += 1
                self.unknowns += 1

                self.scars.append({
                    "context": context,
                    "predicted": None,
                    "observed": actual,
                    "kind": "first_encounter",
                })

            elif prediction == actual:
                line_correct += 1
                self.correct += 1

            else:
                # This is a real model-world mismatch.
                line_errors += 1
                self.errors += 1

                self.scars.append({
                    "context": context,
                    "predicted": prediction,
                    "observed": actual,
                    "kind": "prediction_mismatch",
                })

            if learn:
                # Reality modifies the active structural model.
                self.model[context][actual] += 1

        return {
            "line": line,
            "relations": len(self.contexts(line)),
            "unknowns": line_unknowns,
            "errors": line_errors,
            "correct": line_correct,
        }

    def evaluate(self, lines: list[str]) -> dict[str, float]:
        """
        Evaluate without modifying the model.
        """
        total = 0
        correct = 0
        unknown = 0

        for line in lines:
            for context, actual in self.contexts(line):
                prediction = self.predict(context)
                total += 1

                if prediction is None:
                    unknown += 1
                elif prediction == actual:
                    correct += 1

        accuracy = correct / total if total else 0.0

        return {
            "relations": total,
            "correct": correct,
            "unknown": unknown,
            "accuracy": accuracy,
            "error": 1.0 - accuracy,
        }

    def conceptual_relations(self, lines: list[str]) -> list[dict[str, Any]]:
        """
        Extract explicit conceptual relations such as:

            learning reduces error
            memory changes action
        """
        relation_words = {
            "becomes",
            "creates",
            "changes",
            "improves",
            "preserves",
            "reduces",
            "enables",
            "survives",
            "is",
        }

        relations: list[dict[str, Any]] = []

        for line in lines:
            words = self.tokens(line)

            for i, word in enumerate(words):
                if word not in relation_words:
                    continue

                left = " ".join(words[max(0, i - 2):i])
                right = " ".join(words[i + 1:i + 4])

                if left and right:
                    relations.append({
                        "subject": left,
                        "relation": word,
                        "object": right,
                        "confidence": 0.5,
                    })

        return relations

    def synthesize_program(self) -> None:
        """
       .

        EI writes structure, but never executes generated Python.
        """
        rules = []

        for context, counts in sorted(self.model.items()):
            consequence, evidence = counts.most_common(1)[0]

            rules.append({
                "if_context": context,
                "predict": consequence,
                "evidence": evidence,
                "operation": "predict_next",
            })

        program = {
            "name": "transition_reasoner",
            "language": "EI-safe-dsl-v1",
            "operations": [
                "observe",
                "predict_next",
                "compare",
                "scar_if_wrong",
                "persist",
            ],
            "rules": rules[:1000],
        }

        self.program_path.write_text(
            json.dumps(program, indent=2),
            encoding="utf-8",
        )

    def persist(self) -> None:
        knowledge = []

        for context, counts in sorted(self.model.items()):
            consequence, evidence = counts.most_common(1)[0]

            knowledge.append({
                "context": context,
                "consequence": consequence,
                "evidence": evidence,
            })

        self.tape["knowledge"] = knowledge[:2000]
        self.tape["scars"] = self.scars[-2000:]
        self.tape["program"] = [
            {"op": "observe"},
            {"op": "predict_next"},
            {"op": "compare"},
            {"op": "scar_if_wrong"},
            {"op": "persist"},
        ]
        self.tape["metrics"] = {
            "observations": self.observations,
            "unknowns": self.unknowns,
            "errors": self.errors,
            "correct": self.correct,
            "knowledge_relations": len(knowledge),
            "scar_count": len(self.scars),
        }

        self.tape_path.write_text(
            json.dumps(self.tape, indent=2),
            encoding="utf-8",
        )

    def train(self, lines: list[str]) -> dict[str, Any]:
        for line in lines:
            self.observe(line, learn=True)

        relations = self.conceptual_relations(lines)

        for relation in relations:
            key = f"{relation['subject']}::{relation['relation']}"
            self.model[key][relation["object"]] += 1

        self.synthesize_program()
        self.persist()

        return {
            "lines": len(lines),
            "relations": sum(len(self.contexts(line)) for line in lines),
            "conceptual_relations": len(relations),
            "knowledge_relations": len(self.model),
            "scars": len(self.scars),
        }

    def save_results(self, result: dict[str, Any]) -> None:
        self.results_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )


def split_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    if len(lines) < 4:
        return lines, []

    split = max(1, int(len(lines) * 0.7))
    return lines[:split], lines[split:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EI safe world-model learner"
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--world",
        type=Path,
        default=Path("input/world.txt"),
    )

    parser.add_argument(
        "--transfer",
        type=Path,
        default=Path("input/philosophy.md"),
    )

    args = parser.parse_args()

    root = args.root.resolve()

    world_path = (
        args.world
        if args.world.is_absolute()
        else root / args.world
    )

    transfer_path = (
        args.transfer
        if args.transfer.is_absolute()
        else root / args.transfer
    )

    ei = EI(root)

    world_lines = ei.read_world(world_path)
    train_lines, held_out_lines = split_lines(world_lines)
    transfer_lines = ei.read_world(transfer_path)

    print("=" * 72)
    print("EI — EMERGENT INTELLIGENCE")
    print("=" * 72)
    print(f"Training world : {world_path}")
    print(f"Transfer world : {transfer_path}")
    print(f"Training lines : {len(train_lines)}")
    print(f"Held-out lines : {len(held_out_lines)}")
    print()

    # Evaluate before learning.
    genesis_held_out = ei.evaluate(held_out_lines)
    genesis_transfer = ei.evaluate(transfer_lines)

    print("GENESIS")
    print(f"  held-out accuracy : {genesis_held_out['accuracy']:.3f}")
    print(f"  transfer accuracy : {genesis_transfer['accuracy']:.3f}")
    print()

    # Learn from the training world.
    training_result = ei.train(train_lines)

    learned_held_out = ei.evaluate(held_out_lines)
    learned_transfer = ei.evaluate(transfer_lines)

    print("AFTER LEARNING")
    print(f"  knowledge rules   : {training_result['knowledge_relations']}")
    print(f"  scars             : {training_result['scars']}")
    print(f"  held-out accuracy : {learned_held_out['accuracy']:.3f}")
    print(f"  transfer accuracy : {learned_transfer['accuracy']:.3f}")
    print()

    # Learn from held-out material, then retest transfer.
    ei.train(held_out_lines)
    relearned_transfer = ei.evaluate(transfer_lines)

    print("AFTER RELEARNING")
    print(f"  transfer accuracy : {relearned_transfer['accuracy']:.3f}")
    print()

    result = {
        "genesis": {
            "held_out": genesis_held_out,
            "transfer": genesis_transfer,
        },
        "after_learning": {
            "held_out": learned_held_out,
            "transfer": learned_transfer,
        },
        "after_relearning": {
            "transfer": relearned_transfer,
        },
        "training": training_result,
        "files": {
            "world": str(world_path),
            "transfer": str(transfer_path),
            "tape": str(ei.tape_path),
            "program": str(ei.program_path),
        },
    }

    ei.save_results(result)

    print("PERSISTED")
    print(f"  tape    : {ei.tape_path}")
    print(f"  program : {ei.program_path}")
    print(f"  results : {ei.results_path}")
    print()
    print("INTERPRETATION")
    print("  Unknown is uncertainty, not automatically failure.")
    print("  A mismatch creates a structural scar.")
    print("  Knowledge is reconstructed from the tape.")
    print("  EI writes a safe declarative program.")
    print("  The interpreter remains the external constructor.")


if __name__ == "__main__":
    main()
