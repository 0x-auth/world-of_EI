"""Writes sample_conversation.json: a synthetic two-person chat (720 messages, 4 months) to try convorg on."""
import json, random, datetime as D
random.seed(7)
pairs = [("reached office?", "yes reached, traffic was bad"), ("lunch kiya?", "haan, dal chawal. you?"),
         ("call when free", "in a meeting, will call at 6"), ("movie tonight?", "tired today, weekend chalo"),
         ("did you sleep well", "not really, too much work stress"), ("flight boarded", "have a safe flight, msg when you land"),
         ("landed", "good, take a cab home"), ("happy birthday!", "thank you so much, love the gift"),
         ("gym jaa rahe ho?", "yes, leg day today"), ("dinner ghar pe?", "yes, I'll make khichdi")]
t, out = D.datetime(2024, 1, 1, 9, 0), []
for day in range(120):
    for q, a in random.sample(pairs, 3):
        t += D.timedelta(hours=random.randint(2, 5))
        out.append({"ts": t.isoformat(), "source": "whatsapp", "direction": "sent", "text": q})
        t += D.timedelta(minutes=random.randint(1, 30))
        out.append({"ts": t.isoformat(), "source": "whatsapp", "direction": "received",
                    "text": a if random.random() > .05 else "This message was deleted"})
    t = t.replace(hour=9) + D.timedelta(days=1)
json.dump(out, open("sample_conversation.json", "w"))
print(len(out), "messages -> sample_conversation.json")
