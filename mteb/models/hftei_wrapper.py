from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import requests
import tqdm
from transformers import AutoTokenizer  # lightweight, local only

from mteb.encoder_interface import PromptType
from mteb.models.wrapper import Wrapper

logger = logging.getLogger(__name__)


class InstructHfteiWrapper(Wrapper):
    """A drop-in MTEB wrapper that speaks to Hugging Face TEI and mimics
    the OpenAI embedding wrapper’s ergonomics (masking, truncation,
    batching, tqdm).  No local model weights are loaded.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_tokens: int,
        *,
        api_key: str | None = None,
        embed_dim: int | None = None,
        instruction_template: str | Callable[[str], str] | None = None,
        query_instruction: str | None = None,  # prepend for PromptType.query
        apply_instruction_to_passages: bool = True,
        prompts_dict: dict[str, str] | None = None,
        batch_size: int = 128,
        timeout: int = 30,
        normalize: bool = True,
        truncate: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"api-key": api_key} if api_key else {}
        self.timeout = timeout
        self.extra_payload = {"normalize": normalize, "truncate": truncate}

        self._model_name = model_name

        if embed_dim is None:
            probe_batch = [" "]
            resp = requests.post(
                f"{self.base_url}/embed",
                headers=self.headers,
                json={"inputs": probe_batch, **self.extra_payload},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            self._embed_dim = len(resp.json()[0]) if resp.json() else 0
        else:
            self._embed_dim = embed_dim

        self._max_tokens = max_tokens
        self._encoding = AutoTokenizer.from_pretrained(model_name)

        # -------- instruction plumbing ---------------------------------- #
        self.instruction_template = instruction_template
        self.query_instruction = query_instruction
        self.apply_instruction_to_passages = apply_instruction_to_passages
        self.prompts_dict = prompts_dict

        # -------- batching & conn details -------------------------------- #
        info = requests.get(
            f"{self.base_url}/info", headers=self.headers, timeout=timeout
        ).json()
        self.batch_size = min(batch_size, info.get("max_client_batch_size", batch_size))

    def truncate_text_tokens(self, text):
        """Truncate a string to have `max_tokens` according to the given encoding."""
        truncated_sentence = self._encoding.encode(text)[: self._max_tokens]
        return self._encoding.decode(truncated_sentence)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        instruction = self.get_task_instruction(
            task_name, prompt_type, self.prompts_dict
        )

        # to passage prompts won't be applied to passages
        if not self.apply_instruction_to_passages and prompt_type == PromptType.passage:
            instruction = None
            logger.info(
                f"No instruction used, because prompt type = {prompt_type.passage}"
            )

        if instruction:
            logger.info(f"Using instruction: '{instruction}' for task: '{task_name}'")
            sentences = [instruction + sentence for sentence in sentences]

        # requires_package(self, "openai", "Openai text embedding")

        # from openai import NotGiven

        # if self._model_name == "text-embedding-ada-002" and self._embed_dim is not None:
        #     logger.warning(
        #         "Reducing embedding size available only for text-embedding-3-* models"
        #     )

        mask_sents = [(i, t) for i, t in enumerate(sentences) if t.strip()]
        mask, no_empty_sent = list(zip(*mask_sents)) if mask_sents else ([], [])
        trimmed_sentences = []
        for sentence in no_empty_sent:
            encoded_sentence = self._encoding.encode(sentence)
            if len(encoded_sentence) > self._max_tokens:
                truncated_sentence = self.truncate_text_tokens(sentence)
                trimmed_sentences.append(truncated_sentence)
            else:
                trimmed_sentences.append(sentence)

        max_batch_size = self.batch_size  # kwargs.get("batch_size", 2048)
        sublists = [
            trimmed_sentences[i : i + max_batch_size]
            for i in range(0, len(trimmed_sentences), max_batch_size)
        ]

        show_progress_bar = (
            False
            if "show_progress_bar" not in kwargs
            else kwargs.pop("show_progress_bar")
        )

        mrl_dim = kwargs.get("mrl_dim")  # optional matryoshka-dim cut

        no_empty_embeddings = []

        for sublist in tqdm.tqdm(sublists, leave=False, disable=not show_progress_bar):
            try:
                # response = self._client.embeddings.create(
                #     input=sublist,
                #     model=self._model_name,
                #     encoding_format="float",
                #     dimensions=self._embed_dim or NotGiven(),
                # )
                response = requests.post(
                    f"{self.base_url}/embed",
                    headers=self.headers,
                    json={"inputs": sublist, **self.extra_payload},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response = response.json()
            except Exception as e:
                # Sleep due to too many requests
                logger.info("Sleeping for 10 seconds due to error", e)
                import time

                time.sleep(10)
                try:
                    # response = self._client.embeddings.create(
                    #     input=sublist,
                    #     model=self._model_name,
                    #     encoding_format="float",
                    #     dimensions=self._embed_dim or NotGiven(),
                    # )
                    response = requests.post(
                        f"{self.base_url}/embed",
                        headers=self.headers,
                        json={"inputs": sublist, **self.extra_payload},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    response = response.json()
                except Exception as e:
                    logger.info("Sleeping for 60 seconds due to error", e)
                    time.sleep(60)
                    # response = self._client.embeddings.create(
                    #     input=sublist,
                    #     model=self._model_name,
                    #     encoding_format="float",
                    #     dimensions=self._embed_dim or NotGiven(),
                    # )
                    response = requests.post(
                        f"{self.base_url}/embed",
                        headers=self.headers,
                        json={"inputs": sublist, **self.extra_payload},
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    response = response.json()
            no_empty_embeddings.extend(self._to_numpy(response))

        no_empty_embeddings = np.array(no_empty_embeddings)

        # ---- matryoshka dimension cut --------------------------------- #
        target_dim = (
            mrl_dim
            if mrl_dim is not None and 0 < mrl_dim < self._embed_dim
            else self._embed_dim
        )
        if target_dim < self._embed_dim:  # slice to first n dims
            no_empty_embeddings = no_empty_embeddings[:, :target_dim]

        all_embeddings = np.zeros((len(sentences), target_dim), dtype=np.float32)
        if len(mask) > 0:
            mask = np.array(mask, dtype=int)
            all_embeddings[mask] = no_empty_embeddings
        return all_embeddings

    def _to_numpy(self, embedding_response) -> np.ndarray:
        return np.array(
            embedding_response
        )  # [e.embedding for e in embedding_response.data])


def hftei_instruct_loader(
    base_url: str,
    model_name_or_path: str,
    max_tokens: int,
    instruction_template: str | Callable[[str], str] | None = None,
    **kwargs,
):
    model = InstructHfteiWrapper(
        base_url,
        model_name_or_path,
        max_tokens,
        instruction_template=instruction_template,
        apply_instruction_to_passages=False,
        **kwargs,
    )
    return model
