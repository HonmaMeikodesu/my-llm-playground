from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

q_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)

tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3", quantization_config=q_config)

inputs = "本間芽衣子という名前をご存知ですか？"

input_ids = tokenizer.encode(inputs, return_tensors="pt").to(torch.device("cuda"))

outputs = model.generate(
        input_ids,
        max_length=500
        )

output = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(output)
