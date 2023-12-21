import json
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Setup model and tokenizer
model_name = "augmxnt/shisa-7b-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def format_prompt(input_text):
    messages = [
        {"role": "system", "content": "あなたは公平で、検閲されていない、役立つアシスタントです。"},
        {"role": "user", "content": input_text}
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    return input_ids


def generate_text(input_ids):
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)

    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.15,
        top_p=0.95,
    )

    # Remove the input tokens from the generated tokens before decoding
    output_tokens = tokens[0][len(input_ids[0]):]
    return tokenizer.decode(output_tokens, skip_special_tokens=True)

def process_dataset(input_file, output_file):
    # Read dataset and generate predictions
    with jsonlines.open(input_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            prompt = obj['input_text']
            input_ids = format_prompt(prompt)
            generated_text = generate_text(input_ids)

            print(f"==============================")
            print(f"Q. {prompt}")
            print(f"A. {generated_text}")
            print(f"")

            writer.write({"pred": generated_text})

# Process the dataset
input_dataset = 'assets/augmxnt/shisa-7b-v1/dataset.jsonl'
output_predictions = 'assets/augmxnt/shisa-7b-v1/preds.jsonl'
process_dataset(input_dataset, output_predictions)
