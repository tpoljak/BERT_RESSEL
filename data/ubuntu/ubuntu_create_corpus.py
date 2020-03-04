
SPECIAL_TOK = "[EOT]"

def make_bert_pretrain_data_file(is_eot = True):
  # line by line for training data
  original_path = "/mnt/raid5/taesun/data/Advising/bert_advising_pretrain.txt"
  write_data_path = "/mnt/raid5/taesun/data/ResSel/advising/bert_advising_pretrain.txt"

  dialogue_cnt = 0
  total_cnt = 0
  with open(write_data_path, "w") as fw_handle:
    with open(original_path, "r") as fr_handle:
      for line in fr_handle:
        if line == "\n":
          fw_handle.write('\n')
          dialogue_cnt += 1
          print(dialogue_cnt)
          continue
        total_cnt += 1
        fw_handle.write(line.strip().split(" eot")[0].strip() + '\n')
    # print(total_cnt)

if __name__ == '__main__':
  make_bert_pretrain_data_file()