import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

MODEL_PATH = "nanonets/Nanonets-OCR-s"

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens = 4096):
    prompt = "Extract the text from the above document correctly."
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a OCR model."},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt},]},]
    text = processor.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    inputs = processor(text = [text], images = [image], padding = True, return_tensors = "pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens = max_new_tokens, do_sample = False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens = True, clean_up_tokenization_spaces = True)[0]

def main():
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    image_path = "document.jpeg"
    if not os.path.exists(image_path):
        alt_path = image_path.replace(".jpeg", ".jpg")
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            raise FileNotFoundError("Neither 'document.jpeg' nor 'document.jpg' found.")
    result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=15000)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(result)

if __name__ == "__main__":
    main()

