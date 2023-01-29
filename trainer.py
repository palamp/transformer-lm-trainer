from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, default_data_collator, set_seed
from datasets import load_from_disk
from omegaconf import OmegaConf
import os
import math
import shutil
from utils_v2 import get_result_dir, on_after_train

set_seed(365)


class CustomTrainer(Trainer):
    def __init__(self, config, result_dir: str, **kwargs):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        training_args = TrainingArguments(
            do_train=True, do_eval=True, evaluation_strategy='steps', output_dir=result_dir, **self.config.training_args)
        super().__init__(self.model, training_args, default_data_collator, self._get_train_dataset(),
                         self._get_eval_dataset(), self.tokenizer, **kwargs)

    def _get_train_dataset(self):
        train_dataset = load_from_disk(self.config.data.train_path)
        return train_dataset

    def _get_eval_dataset(self):
        eval_dataset = load_from_disk(self.config.data.test_path)
        return eval_dataset


if __name__ == '__main__':
    conf_file = 'config/config.yaml'
    print(f'Config {conf_file}')
    conf = OmegaConf.load(conf_file)
    result_dir = get_result_dir()
    print('Creating: ', result_dir)
    os.makedirs(result_dir, exist_ok=False)
    shutil.copy(conf_file, f'{result_dir}/config.yaml')
    trainer = CustomTrainer(conf, result_dir)
    train_result = trainer.train()
    trainer.save_model()
    on_after_train(train_result, trainer)
