# Tax Case Data Extraction

An automated pipeline that extracts structured, machine-readable information from tax case judgment PDFs using large language models. Designed to work with cases published by the **[British and Irish Legal Information Institute (BAILII)](https://www.bailii.org/)** — the largest free-access source of British and Irish primary legal materials.

---

## Why This Matters

Tax case judgments are long, complex legal documents — often spanning hundreds of pages — packed with critical information that is difficult and time-consuming to extract manually. Practitioners, researchers, and analysts who need to work across large volumes of case law face a significant bottleneck: reading, summarising, and comparing cases one at a time.

This tool addresses that directly by:

- **Saving hours of manual review** — A case like [TC09685 *Milton Park Holdings Ltd v HMRC* [2025] UKFTT 1353 (TC)](https://www.bailii.org/uk/cases/UKFTT/TC/2025/TC09685.html) runs to over 250 paragraphs of findings of fact, expert evidence, and legal argument. The pipeline extracts all key fields in minutes.
- **Enabling structured comparison** — Once cases are in JSON, they can be queried, filtered, and compared programmatically. Spot trends in judicial reasoning, track how specific legislation is applied, or build knowledge bases.
- **Supporting legal research and due diligence** — Tax advisers, accountants, and litigators can quickly identify relevant precedents, understand the tribunal's reasoning, and assess litigation risk.
- **Powering downstream AI applications** — The structured output feeds directly into RAG (Retrieval-Augmented Generation) pipelines, dashboards, or case management systems.
- **Making public legal data more accessible** — BAILII publishes thousands of judgments that are freely available but locked in unstructured text. Structured extraction opens this corpus to data-driven analysis.

---

## What Gets Extracted

Based on the Pydantic schema defined in [`app/models/schemas.py`](app/models/schemas.py), the pipeline extracts the following structured fields from each judgment:

### `metadata` — Case Identity
| Field | Description | Example (TC09685) |
|---|---|---|
| `case_name` | Full name of the case | *Milton Park Holdings Ltd (1) Milton Park Ltd (2) v HMRC* |
| `neutral_citation` | Neutral citation reference | `[2025] UKFTT 1353 (TC)` |
| `case_number` | Official case number | `TC09685` |
| `court_name` | Court or tribunal name | `First-tier Tribunal (Tax Chamber)` |
| `judgment_date` | Date judgment was handed down | `2025-11-14` |
| `hearing_dates` | List of hearing dates | `[2024-02-05, ..., 2024-02-09]` |
| `judges` | Judge(s) deciding the case | `Tribunal Judge Geraint Williams, Sonia Gable` |
| `parties` | Claimants and respondents | `Milton Park Holdings Ltd (Appellant), HMRC (Respondent)` |
| `representation` | Counsel and firms for each party | `Laurent Sykes KC (Haslers), Michael Jones KC & Harry Winter (HMRC)` |
| `citation_links` | Links to cited legislation | — |

### `facts` — What Happened
| Field | Description |
|---|---|
| `detailed_facts` | Comprehensive chronological summary of background facts and events |
| `key_dates` | Important dates with a description of what occurred on each date |

> In TC09685, this covers the full corporate history from 2004 (formation of the Milton Park Partnership) through to the 2015 sale to Tracscare, including all partnership deeds, restructuring steps, HMRC enquiries opened from 2010, and closure notices issued in 2019.

### `legislation` — Legal Framework
| Field | Description |
|---|---|
| `legislation_list` | All statutes, treaties, and regulations referenced or applied |

> In TC09685: Schedule 29 Finance Act 2002, Part 8 Corporation Tax Act 2009, FRS 5, FRS 6, FRS 10, FRS 11, FRSSE, IFRS 3, FRED 36, s162 TCGA 1992, and more.

### `overview` — Summary
| Field | Description |
|---|---|
| `overview` | Concise summary of the case background, central issues, and context |

### `judges_comments` — Judicial Reasoning
| Field | Description |
|---|---|
| `dicta` | Judicial observations, interpretive principles, and obiter comments |
| `reasoning` | Detailed explanation of how the judges reached their conclusions, including evidence analysis and logical steps |

> In TC09685, this covers the tribunal's analysis of GAAP, substance over form (FRS 5.14), the inseparability of legal goodwill from the business (citing *Muller*, *Star Industrial*, *Geraghty v Minter*), and why MPHL did not acquire control over the operations of MPP.

### `decision` — Outcome
| Field | Description |
|---|---|
| `conclusion` | Final decision (e.g., appeal dismissed, permission refused) |
| `reasoning_summary` | Brief summary of the key reasoning behind the outcome |

> In TC09685: *"The appeals are therefore dismissed."* MPHL did not acquire a business for the purposes of UK GAAP and was not entitled to recognise purchased goodwill.

---

## Sample Case — TC09685

**[Milton Park Holdings Ltd v Revenue and Customs [2025] UKFTT 1353 (TC)](https://www.bailii.org/uk/cases/UKFTT/TC/2025/TC09685.html)**

| | |
|---|---|
| **Issue** | Corporation Tax — whether accounts drawn up in accordance with GAAP — deductions for purchased goodwill amortisation |
| **Court** | First-tier Tribunal (Tax Chamber) |
| **Heard** | 5–9 February 2024 |
| **Decision** | 14 November 2025 |
| **Outcome** | Appeal dismissed |
| **Central question** | Did MPHL acquire control over the *net assets and operations* of the Milton Park Partnership, entitling it to recognise £173m of purchased goodwill? |
| **Result** | No. MCP, not MPHL, operated the business. GAAP (FRS 5/6/FRSSE) requires substance over form. MPHL did not retain the risks and rewards of the business. |

The extracted JSON output for this case can be found in the [`output/`](output/) directory, with results from four different models for comparison:
- [`TC09685_extraction-gpt-4.1.json`](output/TC09685_extraction-gpt-4.1.json)
- [`TC09685_extraction-gpt-4o.json`](output/TC09685_extraction-gpt-4o.json)
- [`TC09685_extraction-gpt-5.json`](output/TC09685_extraction-gpt-5.json)
- [`TC09685_extraction-o3.json`](output/TC09685_extraction-o3.json)

---

## How It Works

```
PDF (BAILII judgment)
        │
        ▼
  Text Extraction          ← PyMuPDF (fitz)
        │
        ▼
  Text Chunking            ← Sliding window, word-based (2000 words, 250 overlap)
        │
        ▼
  LLM Extraction           ← OpenAI via instructor (structured output / Pydantic)
   (per chunk)
        │
        ▼
  Merge & Deduplicate      ← Intelligent field merging across chunks
        │
        ▼
  Validate & Save          ← JSON output to /output directory
```

### Key Design Decisions

- **Chunking with overlap**: Long judgments (TC09685 is ~10,800 lines) exceed model context limits. The pipeline splits text into overlapping word-windows so no information is lost at chunk boundaries.
- **Structured extraction with `instructor`**: Uses [instructor](https://github.com/jxnl/instructor) to enforce Pydantic schema compliance on LLM output, ensuring the JSON always matches the expected schema.
- **Intelligent merging**: After processing each chunk independently, results are merged — lists are extended (deduped), strings are concatenated only if new content is found, and nested dicts are recursively merged.
- **Caching**: Both raw PDF text and extraction results are cached using an MD5 file hash, so re-running the pipeline on the same PDF skips redundant API calls.
- **Retry logic**: API calls use exponential backoff via `tenacity` to handle transient failures gracefully.
- **Model flexibility**: Reasoning models (`o1`, `o3`, `gpt-5`) are detected automatically and the `temperature` parameter is omitted, as these models do not support it.

---

## Project Structure

```
tax-case-data-extraction/
├── main.py                    # Pipeline entry point and all core logic
├── requirements.txt           # Python dependencies
├── .env                       # API keys and configuration (not committed)
├── app/
│   └── models/
│       └── schemas.py         # Pydantic extraction schema
├── docs/                      # Place input PDF judgments here
├── output/                    # Extracted JSON results written here
└── .cache/                    # Auto-generated extraction cache (MD5-keyed)
```

---

## Setup

### Prerequisites
- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Install

```bash
git clone https://github.com/your-org/tax-case-data-extraction.git
cd tax-case-data-extraction
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1          # Optional — defaults to gpt-4.1
USE_CACHE=true                 # Optional — set to false to force re-extraction
PARALLEL_CHUNKS=true           # Optional — enables parallel chunk processing
```

Supported models include any OpenAI chat completion model: `gpt-4o`, `gpt-4.1`, `o3`, `gpt-5`, etc. Reasoning models (`o1`, `o3`, `gpt-5`) are handled automatically.

---

## Usage

### 1. Add PDFs

Download judgment PDFs from [BAILII](https://www.bailii.org/) and place them in the `docs/` directory. For example, the PDF for TC09685 is available at:

```
https://www.bailii.org/uk/cases/UKFTT/TC/2025/TC09685.pdf
```

### 2. Run the pipeline

```bash
python main.py
```

The pipeline will:
1. Auto-discover all `*.pdf` files in `docs/`
2. Extract and cache the text
3. Split into chunks and call the LLM for each
4. Merge, deduplicate, validate, and save results to `output/`

### 3. Process specific files

```python
from main import main

main(
    pdf_files=["docs/TC09685.pdf", "docs/TC09700.pdf"],
    use_cache=True,
    parallel_chunks=True
)
```

### Output

Each processed PDF produces a JSON file in `output/`:

```
output/TC09685_extraction-gpt-4.1.json
```

The JSON mirrors the Pydantic schema exactly, with an additional `_processing_metadata` block:

```json
{
    "metadata": {
        "case_name": "Milton Park Holdings Ltd (1) Milton Park Ltd (2) v HMRC",
        "neutral_citation": "[2025] UKFTT 1353 (TC)",
        "case_number": "TC09685",
        "court_name": "First-tier Tribunal (Tax Chamber)",
        "judgment_date": "2025-11-14",
        "hearing_dates": ["2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08", "2024-02-09"],
        "judges": ["Tribunal Judge Geraint Williams", "Sonia Gable"],
        "parties": [
            "Milton Park Holdings Ltd (Appellant)",
            "Milton Park Ltd (Appellant)",
            "HMRC (Respondent)"
        ],
        "representation": [
            "Laurent Sykes KC (instructed by Haslers Chartered Accountants) for the Appellants",
            "Michael Jones KC and Harry Winter (instructed by HMRC Solicitor) for the Respondents"
        ],
        "citation_links": null
    },
    "facts": {
        "detailed_facts": "...",
        "key_dates": ["1 August 2004 - Milton Park Partnership formed by MM and EM", "..."]
    },
    "legislation": {
        "legislation_list": ["Schedule 29 Finance Act 2002", "Part 8 Corporation Tax Act 2009", "..."]
    },
    "overview": {
        "overview": "..."
    },
    "judges_comments": {
        "dicta": "...",
        "reasoning": "..."
    },
    "decision": {
        "conclusion": "The appeals are therefore dismissed.",
        "reasoning_summary": "..."
    },
    "_processing_metadata": {
        "source_file": "docs/TC09685.pdf",
        "processed_at": "2025-11-20T10:30:00",
        "model": "gpt-4.1",
        "version": "1.0"
    }
}
```

---

## Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1` | Model to use for extraction |
| `USE_CACHE` | `true` | Skip re-processing previously extracted files |
| `PARALLEL_CHUNKS` | `true` | Process text chunks in parallel (up to 3 workers) |

---

## Data Source — BAILII

[BAILII (British and Irish Legal Information Institute)](https://www.bailii.org/) is a registered charity providing free online access to British and Irish legal materials, including:

- **UK First-tier Tribunal (Tax)** — `https://www.bailii.org/uk/cases/UKFTT/TC/`
- **Upper Tribunal (Tax and Chancery)** — `https://www.bailii.org/uk/cases/UKUT/`
- **Court of Appeal** — `https://www.bailii.org/ew/cases/EWCA/`
- **High Court** — `https://www.bailii.org/ew/cases/EWHC/`
- **Supreme Court** — `https://www.bailii.org/uk/cases/UKSC/`

Judgment PDFs are freely downloadable from each case page. This pipeline is designed to process those PDFs directly.

> **Note:** Please respect BAILII's [terms of use](https://www.bailii.org/bailii/copyright.html). This tool is for personal research, professional analysis, and academic use. Do not bulk-scrape BAILII's servers.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pymupdf` (`fitz`) | PDF text extraction |
| `instructor` | Structured LLM output enforced against Pydantic schemas |
| `openai` | OpenAI API client |
| `pydantic` | Data validation and schema definition |
| `python-dotenv` | Environment variable loading |
| `tenacity` | Retry logic with exponential backoff |

See [`requirements.txt`](requirements.txt) for pinned versions.

---

## Extending the Schema

To extract additional fields, edit [`app/models/schemas.py`](app/models/schemas.py). All fields are standard Pydantic `BaseModel` classes with descriptive `Field(...)` annotations that guide the LLM during extraction.

For example, to add a field capturing the tax amount in dispute:

```python
class TaxCaseDecision(BaseModel):
    conclusion: str = Field(...)
    reasoning_summary: Optional[str] = Field(None, ...)
    tax_amount_in_dispute: Optional[str] = Field(
        None,
        description="Total tax amount in dispute, as stated in the judgment."
    )
```

No other code changes are required — `instructor` will automatically populate the new field from the judgment text.

---

## Possible Improvements

### 1. Sentence-based chunking

The current chunker splits text by a fixed word count with a word-level overlap. This means chunks can start or end mid-sentence, which can confuse the LLM and cause information to be dropped or duplicated at boundaries. Replacing the word-based splitter in `split_into_sections()` with a sentence-aware approach — using a library like [`spacy`](https://spacy.io/) or [`nltk.sent_tokenize`](https://www.nltk.org/) — would ensure every chunk begins and ends at a clean sentence boundary, producing more coherent context for the model and reducing noise in the merged output.

### 2. LLM reconciliation pass for narrative fields

The `merge_extraction_data()` function merges long narrative fields (`detailed_facts`, `reasoning`, `dicta`) by simple string concatenation — appending new content if it isn't already a substring of the accumulated value. Across many overlapping chunks, this can produce fragmented or loosely ordered prose. A better approach would be a final reconciliation pass: after all chunks are merged, send the combined draft of each narrative field back to the LLM with a prompt to produce a single coherent, deduplicated summary. This would significantly improve the quality of the `facts` and `judges_comments` sections in particular.

### 3. Parallel multi-PDF processing

The outer loop in `main()` processes each PDF sequentially — one file finishes before the next begins. The `parallel_chunks` flag already enables parallelism *within* a single PDF, but for batch runs across many cases this is a bottleneck. Wrapping the per-file `process_pdf_file()` calls in a `ThreadPoolExecutor` (or `asyncio.gather` with an async client) at the `main()` level would allow multiple PDFs to be processed concurrently, subject to API rate limits.

---

## License

MIT
