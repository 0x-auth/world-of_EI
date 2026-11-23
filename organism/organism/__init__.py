"""EI organism: a frozen rule that learns from its own prediction error, carried as a seed between runs.

Protocol (the pocket card):  Seed -> Medium -> Rule -> Filter -> Seed
  Seed    corrected weights + strikes carried from the last run      (organism/seed.py)
  Medium  any source that can show   events(thread_id, time, role, text)   (organism/sources.py)
  Rule    softmax(b + W^T c), dW = eta * c (y - p)^T, forward + retrocausal twin   (organism/core.py)
  Filter  min/max document frequency, hub lift, strikes read back    (organism/core.py, organism/store.py)
"""
__version__ = "0.1.0"
