from typing import List, Optional, Union

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class GenerationCallback(TrainerCallback):
    def __init__(self, config, result_dir: str) -> None:
        super().__init__()
        self.config = config
        self.result_dir = result_dir
        self.generate_config = self.config.get("generate", {})
        self.log_example: List[str] = self.generate_config.get("examples", [])
        self.generate_preset = self.generate_config.get("generate_preset", "sample")

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
        generate_preset = (
            {
                "no_repeat_ngram_size": 4,
                "do_sample": True,
                "top_p": 0.95,
                "temperature": 0.5,
                "top_k": 4,
                "repetition_penalty": 1.03,
                "penalty_alpha": 0.6,
            }
            if self.generate_preset == "sample"
            else {}
        )
        outputs = model.generate(
            inputs.to(model.device),
            **generate_preset,
            **self.generate_config.get("config", {}),
        )
        decode_text = self._decode_and_remove_newline(tokenizer, outputs)
        return decode_text

    def _decode_and_remove_newline(self, tokenizer, tensor):
        if len(tensor.shape) == 2:
            tensor = tensor[0]
        decode_text = tokenizer.decode(tensor, skip_special_tokens=True)
        remove_newline_decode_text = " ".join(decode_text.split("\n"))
        return remove_newline_decode_text

    def _generate_write_from_example(self, model, tokenizer, example, w):
        generated_text = self._generate(model, tokenizer, example)
        w.write(f"[generation]: {generated_text}\n")

    def _generate_write_from_batch(self, model, tokenizer, dataloader, w, n=1):
        dataloader = iter(dataloader)
        for _ in range(n):
            batch = next(dataloader)
            input_ids = batch["input_ids"][:1, ...]
            labels = batch["labels"][:1, ...]

            inputs_text = self._decode_and_remove_newline(tokenizer, input_ids)
            labels_text = self._decode_and_remove_newline(tokenizer, labels)

            if self.config.get("dataset_type", "clm") == "paraphase_json":
                input_ids = inputs_text.split("Paraphase:")[0] + "Paraphase:"

            generated_text = self._generate(model, tokenizer, input_ids)

            w.write(f"[input]: {inputs_text}\n")
            w.write(f"[gt]: {labels_text}\n")
            w.write(f"[generation]: {generated_text}\n")

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
                os.makedirs(self.result_dir, exist_ok=True)
                write_path = f"{self.result_dir}/examples.txt"
                with open(write_path, "a") as w:
                    w.write(f"=================={state.global_step}==================\n")
                    if len(self.log_example) > 0:
                        for example in self.log_example:
                            self._generate_write_from_example(model, tokenizer, example, w)
                    elif train_dataloader is not None:
                        self._generate_write_from_batch(model, tokenizer, train_dataloader, w)
            except Exception as e:
                print(f"skip generation @ {state.global_step}")
