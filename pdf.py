import os
import fitz
import torch
from PIL import Image, ImageChops
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

MODEL_PATH="nanonets/Nanonets-OCR-s"

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    prompt="Extract the text from the above document correctly."
    image=Image.open(image_path)
    messages=[{"role":"system","content":"You are a OCR model."},{"role":"user","content":[{"type":"image","image":f"file://{image_path}"},{"type":"text","text":prompt}]}]
    text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs=processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    output_ids=model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids=[output_ids[len(input_ids):] for input_ids,output_ids in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

def crop_and_downsize(img_path, output_path, max_side=800):
    img=Image.open(img_path).convert("RGB")
    bg=Image.new("RGB", img.size, (255,255,255))
    diff=ImageChops.difference(img, bg)
    bbox=diff.getbbox()
    if bbox:
        img=img.crop(bbox)
    w,h=img.size
    scale=max_side/max(w,h)
    if scale<1:
        img=img.resize((int(w*scale),int(h*scale)), Image.LANCZOS)
    img.save(output_path)

def main():
    basename=input("Enter PDF basename (without .pdf): ").strip()
    pdf_file=f"{basename}.pdf"
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"{pdf_file} not found.")
    image_folder="images"
    processed_folder="processed"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    doc=fitz.open(pdf_file)
    model=AutoModelForImageTextToText.from_pretrained(MODEL_PATH, torch_dtype="auto", device_map="auto"); model.eval()
    processor=AutoProcessor.from_pretrained(MODEL_PATH)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH)
    output_lines=[]
    for i in range(len(doc)):
        page=doc.load_page(i)
        pix=page.get_pixmap(dpi=150)
        raw_path=os.path.join(image_folder, f"{basename}_page{i+1}.png")
        proc_path=os.path.join(processed_folder, f"{basename}_page{i+1}.png")
        pix.save(raw_path)
        crop_and_downsize(raw_path, proc_path)
        text=ocr_page_with_nanonets_s(proc_path, model, processor, max_new_tokens=15000)
        output_lines.append(f"--- Page {i+1} ---\n{text.strip()}\n")
    out_txt=f"{basename}.txt"
    with open(out_txt,"w",encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"Saved OCR output to {out_txt}")

if __name__=="__main__":
    main()

