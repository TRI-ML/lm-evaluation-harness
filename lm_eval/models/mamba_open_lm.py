import argparse
from typing import List, Optional

import yaml

from lm_eval.api.registry import register_model
from lm_eval.models.open_lm import OpenLMWrapper


@register_model("mamba_open_lm")
class MambaOpenLMWrapper(OpenLMWrapper):
    def __init__(
        self,
        pretrained: str,
        checkpoint: str,
        tokenizer: Optional[str] = "EleutherAI/gpt-neox-20b",
        config_file: str = None,
        **kwargs,
    ) -> None:
        if "backend" in kwargs:
            # mamba currently only supports causal models
            assert kwargs["backend"] == "causal"
        self.config_file = config_file
        self.checkpoint = checkpoint
        super().__init__(
            pretrained,
            checkpoint=checkpoint,
            config_file=config_file,
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
        try:
            from open_lm.model import create_params  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )

        config = self._create_config_dict(pretrained)
        self._config = create_params(config)

    def _create_model(
        self,
        pretrained,
        **kwargs
    ) -> None:
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel  # noqa: F811
            from open_lm.main import load_model # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )

        self._model = MambaLMHeadModel(self._config)

        config = self._create_config_dict(pretrained)

        config.resume = self.checkpoint
        config.distributed = False
        load_strict = False
        if "load_strict"  in kwargs:
            load_strict = bool(kwargs["load_strict"])
        config.load_not_strict = not load_strict
        load_model(config, self._model)
        self._model.eval()