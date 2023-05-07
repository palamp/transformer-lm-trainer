from pathlib import Path

import bitsandbytes as bnb
from omegaconf import DictConfig
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names

from callback import GenerationCallback
from dataset import CloneDataset, EOSSplitTextDataset, JSONDataset
from utils_v2 import is_main_process

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
set_seed(42)


class CustomTrainer(Trainer):
    def __init__(self, config: DictConfig, result_dir: Path):
        self.config = config

        model_config = AutoConfig.from_pretrained(self.config.model.name)
        model_config.gradient_checkpointing = True
        model_config.use_cache = False

        if self.config.model.arch == "enc_dec":
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.name, config=model_config)
        elif self.config.model.arch == "dec_only":
            model = AutoModelForCausalLM.from_pretrained(self.config.model.name, config=model_config).to(
                "cuda"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.get("tokenizer", self.config.model.name)
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<TH_CLS>", "<EN_CLS>"], "cls_token": "<PP_CLS>"}
        )
        model.resize_token_embeddings(len(self.tokenizer))

        self.train_dataset = self._get_train_dataset()
        self.eval_dataset = self._get_eval_dataset()

        training_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            output_dir=result_dir,
            report_to="wandb",
            run_name=f"{self.config.model.name}-{result_dir.name}",
            **self.config.training_args,
        )

        if is_main_process():
            total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total train param: {total_train_params:,}")

        super().__init__(
            model,
            training_args,
            default_data_collator,
            self.train_dataset,
            self.eval_dataset,
            self.tokenizer,
            callbacks=[GenerationCallback(self.config, result_dir)],
            optimizers=(self._get_adam_8bit(model, training_args), None),
            **config.get("trainer_args", {}),
        )

    def _get_adam_8bit(self, model, training_args):
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        return bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            lr=training_args.learning_rate,
        )

    def _get_dataset(self, path: str):
        if self.config.dataset_type == "clm":
            return EOSSplitTextDataset(
                path,
                tokenizer=self.tokenizer,
                arch=self.config.model.arch,
                **self.config.data.get("config", {}),
            )
        elif self.config.dataset_type == "paraphase_json":
            return JSONDataset(
                path,
                tokenizer=self.tokenizer,
                arch=self.config.model.arch,
                **self.config.data.get("config", {}),
            )
        else:
            raise NotImplementedError()

    def _get_train_dataset(self):
        return self._get_dataset(self.config.data.train_path)

    def _get_eval_dataset(self):
        test_path = self.config.data.get("test_path", None)
        if test_path is None:
            self.config.training_args.eval_steps = 10_000_000
            return CloneDataset(self.train_dataset)
        return self._get_dataset(test_path)


if __name__ == "__main__":
    import shutil
    from argparse import ArgumentParser

    from omegaconf import OmegaConf

    from utils_v2 import get_result_dir, on_after_train

    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    args = parser.parse_args()
    conf_file = args.config

    print(f"Config {conf_file}")
    conf = OmegaConf.load(conf_file)
    result_dir = Path(get_result_dir())
    if is_main_process(args.local_rank):
        print("Creating: ", result_dir)
        result_dir.mkdir()
        shutil.copy(conf_file, result_dir / "config.yaml")
    trainer = CustomTrainer(conf, result_dir)
    train_result = trainer.train()
    trainer.save_model()
    on_after_train(trainer, train_result)
