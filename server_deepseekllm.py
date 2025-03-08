from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "Qwen/QwQ-32B"  # Consider switching to "Qwen/Qwen-7B"

# Use 4-bit quantization and enable CPU offload
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True  # Allow CPU offloading
)

# Load model with explicit CPU-GPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="sequential_cpu_offload"  # Gradual offload to CPU
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare prompt
prompt = "How many r's are in the word \"strawberry\""
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response with a safe token limit
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128,  # Lower limit for memory safety
    use_cache=True  # Enable KV caching
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
