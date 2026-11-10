# Metadata Extraction with LangChain, Groq, and Pydantic

Turns unstructured text documents into validated, structured metadata.

## How it works

1. **Extract**: an LLM (Groq, `openai/gpt-oss-20b`) pulls out title, author, publication date, keywords, and document type, using LangChain's `JsonOutputParser`.
2. **Validate**: a Pydantic model checks types and constraints (for example, the date must be a valid date and the keyword list can't be empty).
3. **Route**: valid records go to `metadata.json` and `metadata.csv`. Records that fail validation go to `flagged.json` with the error details and the raw model output.

## Files

| File | Description |
|---|---|
| `extract_metadata.py` | The full pipeline |
| `metadata.json` | Valid records |
| `metadata.csv` | Valid records in CSV form |
| `flagged.json` | Records rejected by validation |

## Sample result

Of three input documents, two passed validation. The third ("A short note with no author, written sometime last spring.") has no title, author, date, or keywords, so the model returned `null` for each field and Pydantic rejected it. It is recorded in `flagged.json`.

## Setup

```bash
pip install langchain langchain-groq langchain-core pydantic
```

Set your Groq API key as an environment variable (never hardcode it):

```powershell
setx GROQ_API_KEY "your_key_here"
```

Open a new terminal, then run:

```bash
python extract_metadata.py
```

Expected output: `2 valid, 1 flagged`

## Notes

- Groq's model catalog changes over time. If a model returns a 404, list the currently available models and update the `model=` line in the script.
- The prompt tells the model to return `null` for anything not stated in the text, so missing information fails validation instead of being invented.
