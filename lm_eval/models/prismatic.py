import argparse
from typing import List, Optional

import os
import yaml
import fsspec
import json
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from prismatic import load

@register_model("prismatic")
class PrismaticVLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        model_id: str,
        tokenizer: Optional[str] = "EleutherAI/gpt-neox-20b",
        **kwargs,
    ) -> None:
        if "backend" in kwargs:
            # mamba currently only supports causal models
            assert kwargs["backend"] == "causal"
        
        super().__init__(
            pretrained,
            backend=kwargs.pop("backend", "causal"),
            tokenizer=tokenizer,
            **kwargs,
        )

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> None:
        with fsspec.open(os.path.join(pretrained, 'config.json')) as f:
          self._config = json.load(f)

    def _create_model(
        self,
        pretrained,
        **kwargs
    ) -> None:
        try:
            from prismatic import load  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )

        vlm = load(pretrained)
        self._model = vlm.llm_backbone.llm
        self._model.model.eval()

