"""Synthetic raw Claude export: 3 accounts, topic chats, plus bridge chats that tie topic words together."""
import json, os, random, sys, uuid

TOPICS = {
    "fabric": "hyperbola mirror twin rapidity invariant fabric boost",
    "primes": "prime sieve gap twin-prime residue modulus",
    "colour": "hex palette rgb saturation hue gradient",
    "relay": "relay proxy token header endpoint gateway",
    "garden": "tomato compost seedling mulch watering soil",
    "music": "raga tabla rhythm scale octave drone",
}
BRIDGES = [("fabric", "colour"), ("primes", "fabric"), ("relay", "garden")]


def msg(rng, words, n=8):
    return " ".join(rng.choice(words) for _ in range(n))


def make(out, seed=7):
    rng = random.Random(seed)
    accts = ["localhost", "phiinverse", "merge"]
    data = {a: [] for a in accts}
    day = 0
    def chat(acct, title, topics, k=8):
        nonlocal day
        day += 1
        words = sum((TOPICS[t].split() for t in topics), [])
        msgs, parent = [], None
        for i in range(k):
            u = str(uuid.UUID(int=rng.getrandbits(128)))
            msgs.append({"uuid": u, "parent_message_uuid": parent, "sender": "human" if i % 2 == 0 else "assistant",
                         "created_at": f"2026-09-{(day % 28) + 1:02d}T10:{i:02d}:00Z",
                         "content": [{"type": "text", "text": msg(rng, words)}]})
            parent = u
        data[acct].append({"uuid": str(uuid.UUID(int=rng.getrandbits(128))), "name": title,
                           "created_at": msgs[0]["created_at"], "updated_at": msgs[-1]["created_at"],
                           "chat_messages": msgs})
    for r in range(3):
        for i, t in enumerate(TOPICS):
            chat(accts[(i + r) % 3], f"{t} talk {r}", [t])
    for r in range(3):
        for a, b in BRIDGES:
            chat(accts[r], f"bridge {a}-{b} {r}", [a, b])
    for a in accts:
        os.makedirs(os.path.join(out, a), exist_ok=True)
        json.dump(data[a], open(os.path.join(out, a, "conversations.json"), "w"))


if __name__ == "__main__":
    make(sys.argv[1])
