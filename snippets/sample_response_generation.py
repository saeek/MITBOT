from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

# Input question (treated as text-to-text)
question = "What is the eligibility for B.Tech?"
input_text = f"question: {question} context: MIT-WPU eligibility criteria"

input_ids = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(input_ids)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Answer:", answer)
