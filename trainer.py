import os

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

from callback import GenerationCallback
from dataset import CloneDataset, EOSSplitTextDataset, JSONDataset
from utils_v2 import is_main_process

# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
set_seed(42)


class CustomTrainer(Trainer):
    def __init__(self, config):
        self.config = config

        model_config = AutoConfig.from_pretrained(self.config.model.name)
        model_config.gradient_checkpointing = True
        model_config.use_cache = False

        if self.config.model.type == "enc_dec":
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.name, config=model_config)
        elif self.config.model.type == "dec_only":
            model = AutoModelForCausalLM.from_pretrained(self.config.model.name, config=model_config)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model.get("tokenizer", self.config.model.name))

        self.train_dataset = self._get_train_dataset()
        self.eval_dataset = self._get_eval_dataset()

        training_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            evaluation_strategy="steps",
            output_dir=result_dir,
            dataloader_num_workers=0,
            learning_rate=self.config.lr,
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
            tokenizer,
            callbacks=[GenerationCallback(self.config, result_dir)],
            **config.get("trainer_args", {}),
        )

    def _get_dataset(self, path: str):
        dataset_type = self.config.get("dataset_type", "clm")
        if dataset_type == "clm":
            return EOSSplitTextDataset(
                path,
                tokenizer_name=self.config.get("tokenizer_name", self.config.model_name),
                arch="clm",
                **self.config.data.get("config", {}),
            )
        elif dataset_type == "paraphase_json":
            return JSONDataset(
                path,
                tokenizer_name=self.config.get("tokenizer_name", self.config.model_name),
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
    result_dir = get_result_dir()
    if is_main_process(args.local_rank):
        print("Creating: ", result_dir)
        os.makedirs(result_dir, exist_ok=False)
        shutil.copy(conf_file, f"{result_dir}/config.yaml")
    trainer = CustomTrainer(conf, result_dir)
    train_result = trainer.train()
    trainer.save_model()
    on_after_train(trainer, train_result)
