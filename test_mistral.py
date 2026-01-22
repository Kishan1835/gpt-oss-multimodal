from model.mistral_wrapper import MistralWrapper

llm = MistralWrapper(device="cuda")
print("Mistral hidden size:", llm.get_hidden_size())
