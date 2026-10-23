"""Step 1: turn the local SQLite copy of digital-history into the time-ordered term stream."""
import os, sys, pickle
import organism as O
DH = os.environ.get("DH_DIR", "data")
db = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DH, "dh.db")
vocab, msgs = O.build(db)
pickle.dump((vocab, msgs), open(os.path.join(DH, "stream.pkl"), "wb"))
print("vocab", len(vocab), "messages", len(msgs))
