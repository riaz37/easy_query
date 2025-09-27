from llama_parse import LlamaParse
import os
from typing import Optional

# Set your Llama Cloud API key once globally
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-5A4BK50a6Uqcjgn1XqEKvV0qX2MNlp17St9RweXCOmWACaIy"

def extract_markdown_from_file(
    file_path: str,
    preserve_layout_alignment_across_pages: bool = True,
    skip_diagonal_text: bool = True,
    output_tables_as_HTML: bool = False,
    disable_image_extraction: bool = False,
    spreadsheet_extract_sub_tables:bool=True,

    result_type: str = "markdown",
) -> str:
    """
    Extracts content from a document file using LlamaParse with configurable options.

    Args:
        file_path (str): Full path to the input document.
        preserve_layout_alignment_across_pages (bool, optional): Preserve layout alignment across pages. Defaults to True.
        skip_diagonal_text (bool, optional): Skip diagonal text in parsing. Defaults to True.
        output_tables_as_HTML (bool, optional): Output tables as HTML if True, else Markdown. Defaults to False.
        disable_image_extraction (bool, optional): Disable image extraction. Defaults to False.
        result_type (str, optional): Output format type. Defaults to "markdown".

    Returns:
        str: Extracted content as a string (markdown, text, or HTML).
    """
    parser = LlamaParse(
        preserve_layout_alignment_across_pages=preserve_layout_alignment_across_pages,
        skip_diagonal_text=skip_diagonal_text,
        output_tables_as_HTML=output_tables_as_HTML,
        disable_image_extraction=disable_image_extraction,
        spreadsheet_extract_sub_tables=spreadsheet_extract_sub_tables,
        result_type=result_type,
    )

    documents = parser.load_data(file_path)

    extracted_content = ""
    for doc in documents:
        extracted_content += doc.text + "\n"

    return extracted_content

# Example usage
if __name__ == "__main__":
    file_path = r"/Users/nilab/Desktop/projects/Knowladge-Base/app/semi-structured/DSE_STOCKS_20230416.xlsx"
    markdown_text = extract_markdown_from_file(
        file_path,
        preserve_layout_alignment_across_pages=False,
        result_type="markdown"
    )

    # Optional: save to file
    with open("output.md", "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(markdown_text)
