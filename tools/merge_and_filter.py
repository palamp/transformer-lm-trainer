# %%
import json

with open('/home/kunato/language-model-agents/instruction_tuning_dataset_alpha_part1.json') as f:
    data = json.load(f)

with open('/home/kunato/language-model-agents/instruction_tuning_dataset_alpha_part2.json') as f:
    data2 = json.load(f)

with open('/home/kunato/instruct-transformer/tools/instruction_tuning_dataset_alpha_part1_th_418.json') as f:
    data3 = json.load(f)
    data3 = list(filter(lambda x: x[0] != 'conala', data3))

with open('/home/kunato/instruct-transformer/tools/instruction_tuning_dataset_alpha_part2_th_418.json') as f:
    data4 = json.load(f)
    data4 = list(filter(lambda x: x[0] != 'conala', data4))
d = data + data2 + data3 + data4


# %%
filter_data = []
for row in d:
    # i dont know why un-consistancy one get better result
    new_row = 'User:' + row[1]
    # new_row = row[1].replace('\n\nRosey:', '\n\n')
    # new_row = 'User:' + new_row.replace('\n\n', '\n\nRosey:')
    filter_data.append([row[0], new_row])

# %%
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(filter_data, test_size=0.01)

# %%
with open('inst_v1_en_th_train.txt', 'w') as w:
    for row in train_df:
        instruction_response_pair = row[1]
        w.write(instruction_response_pair)

with open('inst_v1_en_th_test.txt', 'w') as w:
    for row in test_df:
        instruction_response_pair = row[1]
        w.write(instruction_response_pair)

# %%
