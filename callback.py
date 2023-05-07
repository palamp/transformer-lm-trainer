import json
import re
from pathlib import Path
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class GenerationCallback(TrainerCallback):
    def __init__(self, config, result_dir: Path) -> None:
        super().__init__()
        self.config = config

        self.result_path = result_dir / "examples.json"
        self.result_path.parent.mkdir(exist_ok=True)

        self.generate_config = self.config.get("generate", {})
        self.log_example: List[str] = self.generate_config.get("examples", [])

        if self.generate_config.get("generate_preset", "sample") == "sample":
            self.generation_config = GenerationConfig(
                do_sample=True,
                temperature=0.5,
                top_p=0.95,
                repetition_penalty=1.03,
                no_repeat_ngram_size=4,
                **self.generate_config.get("config", {}),
            )
        else:
            self.generation_config = GenerationConfig(**self.generate_config.get("config", {}))

        self.generate_pattern = re.compile(
            r"(?:<s>)?<TH_CLS>(?P<backTH>.*?)<EN_CLS>(?P<en>.*?)<PP_CLS>(?P<paraphrase>.*?)(?:</s>)*?$"
        )

        print("Will visualize this on going")
        print(self.log_example)

    @torch.no_grad()
    def _generate(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text: Union[str, torch.Tensor],
    ) -> str:
        if isinstance(text, str):
            inputs = tokenizer.encode(text, return_tensors="pt")
        else:
            inputs = text

        outputs = model.generate(inputs.to(model.device), generation_config=self.generation_config)
        decode_text = self._decode_and_remove_newline(tokenizer, outputs)
        return decode_text

    def _decode_and_remove_newline(self, tokenizer, tensor):
        remove_newline_decodes_text = []
        for i, t in enumerate(tensor):
            decode_text = tokenizer.decode(t, skip_special_tokens=False)
            remove_newline_decodes_text.append(" ".join(decode_text.split("\n")))
        return remove_newline_decodes_text

    def _format_dict_result(self, generated_text: List[str], original_text: str):
        dict_text = self.generate_pattern.match(generated_text[0]).groupdict()
        dict_text["original"] = self.generate_pattern.match(original_text).group(3)

        if len(generated_text) > 1:
            dict_text["paraphrase"] = [dict_text["paraphrase"]]
            for gen_text in generated_text[1:]:
                dict_text["paraphrase"].append(self.generate_pattern.match(gen_text).group(3))

        return dict_text

    def _generate_write_from_example(self, model, tokenizer):
        example_result = []
        for example in self.log_example:
            generated_text = self._generate(model, tokenizer, example.split("<PP_CLS>")[0] + "<PP_CLS>")
            example_result.append(self._format_dict_result(generated_text, example))
        return example_result

    def _generate_write_from_batch(self, model, tokenizer, dataloader, n=1):
        batch_result = []
        dataloader = iter(dataloader)
        for _ in range(n):
            batch = next(dataloader)

            inputs_text = self._decode_and_remove_newline(tokenizer, batch["input_ids"][:1, ...])[0]
            input_ids = inputs_text.split("<PP_CLS>")[0] + "<PP_CLS>"

            generated_text = self._generate(model, tokenizer, input_ids)
            batch_result.append(self._format_dict_result(generated_text, inputs_text))
        return batch_result

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataloader: Optional[DataLoader] = None,
        **kwargs,
    ):
        if state.is_local_process_zero:
            try:
                if len(self.log_example) > 0:
                    output = self._generate_write_from_example(model, tokenizer)
                elif train_dataloader is not None:
                    output = self._generate_write_from_batch(model, tokenizer, train_dataloader)

                with open(self.result_path, "a") as w:
                    w.write(json.dumps({state.global_step: output}, ensure_ascii=False, indent=4))
            except Exception as e:
                print(f"skip generation {state.global_step} with error {e.args}")


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dataset import JSONDataset

    config = OmegaConf.load("config/config_paraphrase_base.yaml")
    callback = GenerationCallback(config, Path())

    model = AutoModelForCausalLM.from_pretrained("facebook/xglm-564M")
    tokenizer = AutoTokenizer.from_pretrained("facebook/xglm-564M")
    dataset = JSONDataset("data/test_paraphrase_v2.json", tokenizer, **config.data.get("config", {}))

    callback.on_log(
        TrainingArguments(Path()),
        TrainerState(),
        TrainerControl(),
        model,
        tokenizer,
        DataLoader(dataset, shuffle=False),
    )
