# %%
import random


def main(dup_factor=10):
    with open('filelist/v3/deeple.txt') as f:
        data = f.read()
        deeple_data = data.split('<|endoftext|>')
        total_data = []
        for i in range(dup_factor):
            total_data += deeple_data
    with open('filelist/v2/inst_v1_th_en_train.txt') as f:
        data = f.read()
        reg_data = data.split('<|endoftext|>')
        need = len(total_data)
        total_data += reg_data[:need]
    random.shuffle(total_data)
    with open('filelist/v3/deeple_reg.txt', 'w') as w:
        for row in total_data:
            w.write(f'{row}<|endoftext|>')


if __name__ == '__main__':
    main()
