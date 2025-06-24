#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from sentence_transformers.util import cos_sim

import mteb
from mteb.encoder_interface import PromptType

# ------------------------------------------------------------
# 1 · Load both models (same checkpoint, different wrappers)
# ------------------------------------------------------------
local_model = mteb.get_model("Qwen/Qwen3-Embedding-0.6B")
tei_model = mteb.get_model("hftei/Qwen/Qwen3-Embedding-0.6B")

# ------------------------------------------------------------
# 2 · Example corpus: 2 queries + 2 passages
# ------------------------------------------------------------
queries = [
    "Wie heißt die Hauptstadt von Deutschland?",
    "Wann wurde die Berliner Mauer gebaut?",
]
passages = [
    "Berlin ist die Hauptstadt Deutschlands.",
    "Die Berliner Mauer wurde 1961 errichtet, um Ost- und West-Berlin zu trennen.",
]

task_name = "GermanQuAD-Retrieval"

# ------------------------------------------------------------
# 3 · Encode with the right prompt_type
# ------------------------------------------------------------
q_kwargs = dict(task_name=task_name, prompt_type=PromptType.query, batch_size=4)
p_kwargs = dict(task_name=task_name, prompt_type=PromptType.passage, batch_size=4)

loc_q = local_model.encode(queries, **q_kwargs)
loc_p = local_model.encode(passages, **p_kwargs)
tei_q = tei_model.encode(queries, **q_kwargs)
tei_p = tei_model.encode(passages, **p_kwargs)


# ------------------------------------------------------------
# 4 · Helper: check normalisation + print norms
# ------------------------------------------------------------
def norms(v):
    return np.linalg.norm(v, axis=1)


print("-- vector ℓ2-norms --")
print("local query   :", np.round(norms(loc_q), 4))
print("tei   query   :", np.round(norms(tei_q), 4))
print("local passage :", np.round(norms(loc_p), 4))
print("tei   passage :", np.round(norms(tei_p), 4))
print()

# Are they already unit-length?
print(
    "local normalised? ",
    np.allclose(norms(loc_q), 1, atol=1e-3) and np.allclose(norms(loc_p), 1, atol=1e-3),
)
print(
    "tei   normalised? ",
    np.allclose(norms(tei_q), 1, atol=1e-3) and np.allclose(norms(tei_p), 1, atol=1e-3),
)
print()

# ------------------------------------------------------------
# 5 · Cosine(local, tei) per example
# ------------------------------------------------------------
q_sim = cos_sim(loc_q, tei_q).diagonal()
p_sim = cos_sim(loc_p, tei_p).diagonal()

print("-- cosine(local ↔ tei) --")
for i, s in enumerate(q_sim):
    print(f"query  {i + 1}: {s:.4f}")
for i, s in enumerate(p_sim):
    print(f"passage {i + 1}: {s:.4f}")

all_sim = np.concatenate([q_sim, p_sim])
print("\nmean:", all_sim.mean(), "min:", all_sim.min())
