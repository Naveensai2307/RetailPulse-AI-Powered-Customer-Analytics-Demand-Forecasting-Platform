import pypdf
import sys
import os

def extract_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Path to the instructions folder relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(script_dir, "..", "instructions")
    
    # Process all PDFs in the instructions folder
    if os.path.exists(pdf_dir):
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
        for pdf_name in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_name)
            print(f"Extracting from: {pdf_name}...")
            content = extract_text(pdf_path)
            
            output_name = f"{os.path.splitext(pdf_name)[0]}_content.txt"
            output_path = os.path.join(script_dir, "..", output_name)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Success! Content saved to {output_name}")
    else:
        print(f"Error: Folder '{pdf_dir}' not found.")
