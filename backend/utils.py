import PyPDF2
import docx

def extract_text(file_path):

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif file_path.endswith(".pdf"):
        text = ""
        reader = PyPDF2.PdfReader(file_path)

        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

        return text

    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    return ""