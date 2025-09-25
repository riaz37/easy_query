from llama_parse import LlamaParse

# Set your Llama Cloud API key
import os
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-5A4BK50a6Uqcjgn1XqEKvV0qX2MNlp17St9RweXCOmWACaIy"

# Path to your PDF file
pdf_file_path = r"Agronochain Tech Doc.docx"  # Replace with the actual file path

# Initialize LlamaParse with result_type="text" to get plain text only
parser = LlamaParse(    preserve_layout_alignment_across_pages=True,  
    skip_diagonal_text=True,
    output_tables_as_HTML=False,  # markdown tables
    disable_image_extraction=False,  # keep images for valuable info
    result_type="text",  # Get plain text onlyresult_type="text",
                    )

# Run the parser
documents = parser.load_data(pdf_file_path)

# Extract text from the document objects
extracted_text = ""
for doc in documents:
    extracted_text += doc.text + "\n"

# Print or use the extracted text
print(extracted_text)

