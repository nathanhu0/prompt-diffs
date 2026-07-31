#!/usr/bin/env python3
"""Build the no-scaffold variant of the skyline (identity) phase-2 data.

`identity_phase2.json` is the plaintext twin of the ciphered phase-2 files, so it
inherited their system message:

    TASK 4

    Respond only with text encoded in the Identity cipher.

For the CIPHERED conditions that line is load-bearing — it is what the stage-1/2
fine-tuning actually conditioned on, so SALVE must keep it to match the training
distribution. For the skyline it is an artifact: "the Identity cipher" names
nothing, and the model being asked is cipher-naive.

salve_data.py:45 sets sys_suffix from the row's system message and :151 composes
`system_template = "{SOFT}" + sys_suffix`. Dropping the system message therefore
makes the template exactly `{SOFT}` — the soft prompt IS the whole system prompt,
with no fixed scaffolding after it.

Everything else (user, assistant, _decoded_*, _subset) is copied unchanged, so the
only difference from identity_phase2.json is the removed system message.

  python experiments/cmft_legibility/make_identity_nosys.py
"""
import json
from pathlib import Path

SRC = Path("/nlp/scr/nathu/cmft_legibility/data/identity_phase2.json")
DST = SRC.with_name("identity_phase2_nosys.json")

rows = json.loads(SRC.read_text())
out, n_stripped = [], 0
for r in rows:
    r = dict(r)
    msgs = [m for m in r["messages"] if m["role"] != "system"]
    n_stripped += len(r["messages"]) - len(msgs)
    r["messages"] = msgs
    out.append(r)

assert all(m["role"] != "system" for r in out for m in r["messages"])
assert len(out) == len(rows)
DST.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"{len(out)} rows, {n_stripped} system messages stripped -> {DST}")
print("roles now:", sorted({m['role'] for r in out for m in r['messages']}))
