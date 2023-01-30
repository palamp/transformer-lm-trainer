import glob
import math
import os


def get_result_dir(lightning_logs_dir='results'):
    log_dir = glob.glob(f'{lightning_logs_dir}/**/config.yaml')
    log_dir_version = list(
        map(lambda x: int(x.split('/')[-2].replace('version_', '')), log_dir))
    if len(log_dir_version) != 0:
        exp_num = '{:0>3d}'.format(max(log_dir_version) + 1)
    else:
        exp_num = '{:0>3d}'.format(1)
    return f'./results/{exp_num}'


def on_after_train(trainer, train_result):
    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer._get_train_dataset())
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(trainer._get_eval_dataset())
    perplexity = math.exp(metrics["eval_loss"])
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def is_main_process(env_local_rank=None):
    if env_local_rank is None:
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    return env_local_rank == 0 or env_local_rank == -1
