import argparse
from typing import List, Optional

import yaml
import fsspec
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("open_lm")
class OpenLMWrapper(HFLM):
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
            from open_lm.utils.transformers.hf_config import OpenLMConfig  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )

        config = self._create_config_dict(pretrained)
        self._config = OpenLMConfig(create_params(config))

    def _create_model(
        self,
        pretrained,
        **kwargs
    ) -> None:
        try:
            from open_lm.utils.transformers.hf_model import OpenLMforCausalLM  # noqa: F811
            from open_lm.main import load_model, pt_load # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )

        self._model = OpenLMforCausalLM(self._config)

        config = self._create_config_dict(pretrained)

        config.resume = self.checkpoint
        config.distributed = False
        load_strict = False
        if "load_strict"  in kwargs:
            load_strict = bool(kwargs["load_strict"])
        config.load_not_strict = not load_strict
        checkpoint = pt_load(config.resume, map_location="cpu")
        if "llm" in checkpoint:
            checkpoint = checkpoint["llm"]
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if "llm_backbone" in checkpoint:
            checkpoint = checkpoint["llm_backbone"]
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        checkpoint = {k: v for k, v in checkpoint.items() if "inv_freq" not in k}
        self._model.model.load_state_dict(checkpoint, strict=load_strict)
        self._model.model.eval()

    def _create_config_dict(self, pretrained: str, **kwargs) -> None:
        try:
            from open_lm.params import add_model_args#, add_training_args  # noqa: F811
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'open_lm' LM type, but package `open_lm` is not installed." \
                "please install open_lm from `https://github.com/TRI-ML/open_lm`",
            )
        parser = argparse.ArgumentParser()
        # add_training_args(parser)
        add_model_args(parser)

        config = parser.parse_args([])
        config.model = pretrained

        if self.config_file is not None:
            with fsspec.open(self.config_file) as f:
                config_to_override = yaml.safe_load(f)
            for k, v in config_to_override.items():
                if v == "None":
                    v = None

                # we changed args
                if k == "batch_size":
                    k = "per_gpu_batch_size"
                if k == "val_batch_size":
                    k = "per_gpu_val_batch_size"
                setattr(config, k, v)
        return config