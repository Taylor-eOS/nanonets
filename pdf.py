import os
import fitz
import torch
from PIL import Image, ImageChops
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import settings

def ocr_page(image_path, model, processor, max_new_tokens=settings.MAX_TOKENS):
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": settings.MODEL_DESCR}, 
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"}, 
            {"type": "text", "text": settings.PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

def crop_image(img, margin=settings.CROP_MARGIN):
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        left, upper, right, lower = bbox
        left = max(left - margin, 0)
        upper = max(upper - margin, 0)
        right = min(right + margin, img.width)
        lower = min(lower + margin, img.height)
        img = img.crop((left, upper, right, lower))
    return img

def resize_image(img, max_side=settings.MAX_PAGE_HEIGHT):
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def prepare_folders(*folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def load_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, processor, tokenizer

def process_pdf(doc, basename, image_folder, processed_folder, model, processor):
    output_lines = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=settings.IMAGE_QUALITY)
        raw_path = os.path.join(image_folder, f"{basename}_page{i+1}.png")
        proc_path = os.path.join(processed_folder, f"{basename}_page{i+1}.png")
        pix.save(raw_path)
        raw = Image.open(raw_path).convert("RGB")
        cropped = crop_image(raw)
        resized = resize_image(cropped)
        resized.save(proc_path)
        text = ocr_page(proc_path, model, processor)
        page_number = f"---Page {i+1}---\n"
        output_lines.append(f"{page_number}{text.strip()}\n")
    return output_lines

def main():
    basename = input("PDF basename: ").strip()
    pdf_file = f"{basename}.pdf"
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"{pdf_file} not found.")
    image_folder = "images"
    processed_folder = "images/processed"
    prepare_folders(image_folder, processed_folder)
    doc = fitz.open(pdf_file)
    model, processor, tokenizer = load_model(settings.MODEL_PATH)
    output_lines = process_pdf(doc, basename, image_folder, processed_folder, model, processor)
    out_txt = f"{basename}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"Saved OCR output to {out_txt}")

if __name__=="__main__":
    main()

