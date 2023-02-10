# %%
import random


def main():
    bot_name = "Deeple"
    instruction = f'The following is conversation between an Deeple assistant called {bot_name}, and a human user called User. {bot_name} is intelligent, knowledgeable, wise and polite.'
    with open('filelist/v2/inst_v1_th_en_train.txt') as f:
        data = f.read()
        deeple_data = data.split('<|endoftext|>')
        total_data = []
        for d in deeple_data:
            total_data.append(f'{instruction}\n\n{d}')
    with open('filelist/v2/inst_v1_th_en_train_instruct.txt', 'w') as w:
        for row in total_data:
            w.write(f'{row}<|endoftext|>')

    with open('filelist/v2/inst_v1_th_en_test.txt') as f:
        data = f.read()
        deeple_data = data.split('<|endoftext|>')
        total_data = []
        for d in deeple_data:
            total_data.append(f'{instruction}\n\n{d}')

    with open('filelist/v2/inst_v1_th_en_test_instruct.txt', 'w') as w:
        for row in total_data:
            w.write(f'{row}<|endoftext|>')


if __name__ == '__main__':
    main()
