#!/usr/bin/env python3
from __future__ import annotations

import mteb

model = mteb.get_model("hftei/Qwen/Qwen3-Embedding-0.6B")

tasks = mteb.get_tasks(
    tasks=["GermanQuAD-Retrieval", "GermanDPR"]  # , "XMarket", "GerDaLIR"]
)
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(
    model,
    verbosity=2,
    overwrite_results=True,
    # encode_kwargs={
    #     "batch_size": 1
    # },
)

for result in results:
    print(f"{result.task_name} {result.scores['test'][0]['main_score']}")

# Results stored in ./results
