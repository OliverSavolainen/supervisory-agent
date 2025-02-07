import os
from fpdf import FPDF
from crewai_tools import tool
from docx import Document

@tool("File writer")
def file_writer_tool(file_name: str, content: str) -> str:
    """"Useful for writing text content into a document format. Supports pdf, docx and txt. Requires file_name and content"""
    _, file_extension = os.path.splitext(file_name)
    try:
        if file_extension.lower() == '.pdf':
            write_pdf(file_name, content)
        #elif file_extension.lower() == '.docx':
         #   write_docx(file_name, content)
        elif file_extension.lower() == '.txt':
            write_txt(file_name, content)
        else:
            return f"Error: Unsupported file type: {file_extension}"
        return f"Successfully wrote content to {file_name}. Task complete."
    except Exception as e:
        return f"An error occurred: {str(e)}"


def write_pdf(file_name, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, content)
    pdf.output(file_name)


def write_docx(file_name, content):
    doc = Document()
    doc.add_paragraph(content)
    doc.save(file_name)


def write_txt(file_name, content):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(content)
