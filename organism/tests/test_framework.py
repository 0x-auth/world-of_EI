import json, os, sqlite3, subprocess, sys, glob
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from make_fixture import make  # noqa: E402

ARGS = ["--min-df", "2", "--max-df", "0.5"]


def run(*a):
    return subprocess.run([sys.executable, "-m", "organism", *a], check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx")
    make(str(d / "raw"))
    run("clean", str(d / "raw"), str(d / "clean"))
    return d


def folders(d):
    return [str(d / "clean" / a) for a in ("localhost", "phiinverse", "merge")]


def test_same_links_as_v1_4(data):
    subprocess.run([sys.executable, os.path.join(HERE, "reference", "organism_v1_4.py"), *folders(data),
                    "--focus", "", *ARGS, "--out", str(data / "ref")], check=True, capture_output=True)
    run("run", *folders(data), *ARGS, "--parts", "0", "--out", str(data / "fw"), "--store", str(data / "fw.db"), "--seed", "none")
    r = json.load(open(data / "ref" / "links.json")); f = json.load(open(data / "fw" / "links.json"))
    assert [(x["a"], x["b"]) for x in r] == [(x["a"], x["b"]) for x in f]
    assert all(abs(x["hidden"] - y["hidden"]) < 1e-9 for x, y in zip(r, f))


def test_strike_is_not_reborn_and_seed_is_carried(data):
    db = str(data / "s.db")
    run("run", *folders(data), *ARGS, "--parts", "0", "--out", str(data / "s1"), "--store", db, "--seed", "none")
    rid = sqlite3.connect(db).execute("select run_id from organism_runs").fetchone()[0]
    first = json.load(open(data / "s1" / "links.json"))[0]
    u, v, _ = first["carried_by"][0]
    run("strike", rid, "1", "--store", db)
    import time; time.sleep(1.1)
    out = run("run", *folders(data), *ARGS, "--parts", "0", "--out", str(data / "s2"), "--store", db)
    assert "carried seed" in out
    links = json.load(open(data / "s2" / "links.json"))
    assert not any({a, b} == {u, v} for l in links for a, b, _ in l["carried_by"])


def test_any_database_sqlite_view(data):
    src = str(data / "src.db"); con = sqlite3.connect(src)
    con.execute("create table m(c, t, r, x)"); con.execute("create table c(id, n, a, u)")
    for p in glob.glob(str(data / "clean" / "*" / "conversations.json")):
        for c in json.load(open(p)):
            con.execute("insert into c values(?,?,?,?)", (c["uuid"], c["name"], c["account"], c["updated_at"]))
            for m in c["chat_messages"]:
                con.execute("insert into m values(?,?,?,?)", (c["uuid"], m["created_at"], m["sender"], m["text"]))
    con.execute("create view events as select c thread_id, t time, r role, x text from m")
    con.execute("create view threads as select id thread_id, n title, a account, u updated from c")
    con.commit()
    run("run", f"sqlite:{src}", *ARGS, "--parts", "0", "--out", str(data / "sq"), "--store", str(data / "sq.db"), "--seed", "none")
    a = json.load(open(data / "sq" / "links.json")); b = json.load(open(data / "fw" / "links.json"))
    assert sorted((x["a"], x["b"]) for x in a) == sorted((x["a"], x["b"]) for x in b)


def test_parts_point_into_threads(data):
    run("run", *folders(data), *ARGS, "--parts", "4", "--out", str(data / "pt"), "--store", str(data / "pt.db"), "--seed", "none")
    links = json.load(open(data / "pt" / "links.json"))
    assert links and all(l["a"] != l["b"] and l["a_msgs"][0] >= 1 and l["a_msgs"][1] >= l["a_msgs"][0] for l in links)


def test_gap_split_sessions():
    from organism import core
    c = [{"id": "w", "acct": "wa", "title": "WA", "start": "2020-01-01T10:00:00Z", "updated": "",
          "msgs": [("2020-01-01T10:00:00Z", "u", "a"), ("2020-01-01T11:00:00Z", "u", "b"),
                   ("2020-01-03T09:00:00Z", "u", "c"), ("2020-01-03T09:05:00", "u", "d")]},
         {"id": "x", "acct": "a", "title": "X", "start": "2020-01-02", "updated": "", "msgs": [("2020-01-02", "u", "z")]}]
    s = core.split_gaps(c, 6)
    assert [t["id"] for t in s] == ["w#0", "x", "w#1"] and [len(t["msgs"]) for t in s] == [2, 1, 2]
    assert s[2]["title"] == "WA · 2020-01-03"


def test_sessions_of_one_thread_do_not_link():
    import numpy as np
    from organism import core
    w = "alpha beta gamma delta epsilon zeta eta theta".split()
    mk = lambda d, k: [(f"2020-01-{d:02d}T{h:02d}:00:00Z", "u", " ".join(w[(h + k) % 8:(h + k) % 8 + 3])) for h in range(8)]
    chats = core.split_gaps([{"id": "w", "acct": "a", "title": "W", "start": "", "updated": "", "msgs": mk(1, 0) + mk(5, 1)},
                             {"id": "x", "acct": "b", "title": "X", "start": "", "updated": "", "msgs": mk(3, 2)}], 6)
    vocab = sorted(set(w)); L = np.ones((8, 8), np.float32); np.fill_diagonal(L, 0)
    links = core.part_links(chats, vocab, L, size=4, top=10)
    assert links and all({l["a"].split("#")[0], l["b"].split("#")[0]} == {"w", "x"} for l in links)
