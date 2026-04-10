import fitz
import os
from pathlib import Path

def pdf_to_images(pdf_path, output_folder):
    """
    Converts each page of a PDF to a PNG image.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    doc = fitz.open(pdf_path)
    base_name = Path(pdf_path).stem
    
    for i, page in enumerate(doc):
        # 300 DPI for high quality layout detection
        pix = page.get_pixmap(dpi=300)
        output_path = os.path.join(output_folder, f"{base_name}_page_{i}.png")
        pix.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    drawings_dir = Path("drawings")
    output_dir = Path("images")
    
    # Process all PDFs in drawings folder
    pdf_files = list(drawings_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")
    
    # Let's process the first 50 as recommended (20-50 PDFs minimum)
    for pdf_file in pdf_files[:len(pdf_files)]:
        print(f"Processing {pdf_file}...")
        pdf_to_images(str(pdf_file), str(output_dir))
