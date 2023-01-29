from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, set_seed
from dataset.dataset import EOSSplitTextDataset
from omegaconf import OmegaConf
import os
import shutil
from utils_v2 import get_result_dir, on_after_train, is_main_process

set_seed(365)


class CustomTrainer(Trainer):
    def __init__(self, config, result_dir: str, **kwargs):
        self.config = config
        self.is_enc_dec = 't5' in self.config.model_name
        if self.is_enc_dec:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name)
        self._set_tokenizer()
        training_args = TrainingArguments(
            do_train=True, do_eval=True, evaluation_strategy='steps', output_dir=result_dir, dataloader_num_workers=8, **self.config.training_args)
        super().__init__(self.model, training_args, default_data_collator, self._get_train_dataset(),
                         self._get_eval_dataset(), self.tokenizer, **kwargs)

    def _set_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if 'pythia' in self.config.model_name:
            # neox / decoder only doesnt' have pad token id, but there are in vocab, but not used
            self.tokenizer.pad_token_id = 1

    def _get_train_dataset(self):
        data_config = self.config.data
        train_dataset = EOSSplitTextDataset(
            data_config.train_path, self.tokenizer, arch='prefix_lm' if self.is_enc_dec else 'clm', **data_config.get('config', {}))
        return train_dataset

    def _get_eval_dataset(self):
        data_config = self.config.data
        eval_dataset = EOSSplitTextDataset(
            data_config.test_path, self.tokenizer, arch='prefix_lm' if self.is_enc_dec else 'clm', **data_config.get('config', {}))
        return eval_dataset


if __name__ == '__main__':
    conf_file = 'config/config_clm.yaml'
    print(f'Config {conf_file}')
    conf = OmegaConf.load(conf_file)
    result_dir = get_result_dir()
    if is_main_process():
        print('Creating: ', result_dir)
        os.makedirs(result_dir, exist_ok=False)
        shutil.copy(conf_file, f'{result_dir}/config.yaml')
    trainer = CustomTrainer(conf, result_dir)
    train_result = trainer.train()
    trainer.save_model()
    on_after_train(train_result, trainer)
