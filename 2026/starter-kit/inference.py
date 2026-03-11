from unsloth import FastVisionModel
from datasets import load_dataset

def main():
    print("Loading model for inference...")
    model, tokenizer = FastVisionModel.from_pretrained(
        "lora_model", # Path to saved adapter
        load_in_4bit = True,
    )
    FastVisionModel.for_inference(model) # Enable native 2x faster inference
    
    print("Loading validation dataset...")
    dataset = load_dataset("MBZUAI/EXAMS-V", split="validation", streaming=True) # Streaming to avoid full download if huge
    
    # Take first 5 examples
    print("Running inference on 5 examples...")
    for i, example in enumerate(dataset.take(5)):
        image = example['image']
        # question = example['question'] # Not available
        answer = example['answer_key']
        
        prompt_text = "Answer the multiple choice question presented in the image. Return the letter corresponding to the correct answer."
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Prepare inputs
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
        
        # Process images and text
        from qwen_vl_utils import process_vision_info
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate
        outputs = model.generate(**inputs, max_new_tokens=128)
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n--- Example {i+1} ---")
        print(f"Question: [Image]")
        print(f"Correct Answer: {answer}")
        print(f"Model Output: {response}")

if __name__ == "__main__":
    main()
