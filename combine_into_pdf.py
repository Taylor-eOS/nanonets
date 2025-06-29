from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def split_into_pages(lines):
    pages = []
    current_page = []
    for line in lines:
        if line.strip().startswith('---Page ') and line.strip().endswith('---'):
            if current_page:
                pages.append(current_page)
                current_page = []
        else:
            current_page.append(line.rstrip())
    if current_page:
        pages.append(current_page)
    return pages

def write_pdf(output_pdf, pages):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    for page_lines in pages:
        y = height - 50
        for line in page_lines:
            c.drawString(50, y, line)
            y -= 14
            if y < 50:
                c.showPage()
                y = height - 50
        c.showPage()
    c.save()

if __name__ == "__main__":
    basename = input("File basename: ").strip()
    if not basename:
        raise ValueError("No basename provided.")
    txt_file = f"{basename}.txt"
    with open(txt_file, encoding="utf-8") as f:
        lines = f.readlines()
    pages = split_into_pages(lines)
    output_pdf = f"{basename}_output.pdf"
    write_pdf(output_pdf, pages)
    print(f"Wrote PDF to {output_pdf}")

