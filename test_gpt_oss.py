from model.gpt_oss_wrapper import GPTOSSWrapper

gpt = GPTOSSWrapper(device="cpu")
print("GPT-OSS hidden size:", gpt.get_hidden_size())
