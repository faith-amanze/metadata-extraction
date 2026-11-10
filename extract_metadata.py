import json
import csv
from datetime import date

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError


class DocumentMetadata(BaseModel):
    title: str = Field(..., min_length=1, description="Title of the document")
    author: str = Field(..., min_length=1, description="Author of the document")
    publication_date: date = Field(..., description="Date in YYYY-MM-DD format")
    keywords: list[str] = Field(..., min_length=1, description="List of keywords")
    document_type: str = Field(..., description="Type of the document, e.g., report, article")


parser = JsonOutputParser(pydantic_object=DocumentMetadata)

prompt = PromptTemplate(
    template="""Extract the following metadata from the document text:
title, author, publication_date, keywords, and document_type.

Only use information stated in the text. If a field is not stated,
return null for it. Do not guess or invent values.

{format_instructions}

Document text:
{document_text}
""",
    input_variables=["document_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
chain = prompt | model | parser

documents = [
    """This report, 'AI Trends 2025', was written by Dr. Sarah Lee and published on 2025-05-01.
       It covers topics like artificial intelligence, deep learning, and ethics in AI.""",
    """The article 'Climate Change and Agriculture' by John Smith was released on 2024-09-15.
       It discusses sustainability, farming practices, and environmental policy.""",
    """A short note with no author, written sometime last spring.""",
]

valid, flagged = [], []

for idx, doc in enumerate(documents, start=1):
    raw = None
    try:
        raw = chain.invoke({"document_text": doc})
        record = DocumentMetadata(**raw)
        valid.append(record.model_dump(mode="json"))
    except ValidationError as e:
        flagged.append({"doc_index": idx, "error": e.errors(), "raw": raw})
    except Exception as e:
        flagged.append({"doc_index": idx, "error": str(e)})

with open("metadata.json", "w") as f:
    json.dump(valid, f, indent=2)

with open("metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=DocumentMetadata.model_fields.keys())
    writer.writeheader()
    for row in valid:
        row["keywords"] = "; ".join(row["keywords"])
        writer.writerow(row)

with open("flagged.json", "w") as f:
    json.dump(flagged, f, indent=2, default=str)

print(f"{len(valid)} valid, {len(flagged)} flagged")