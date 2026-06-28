# India Runs Track 1

A hybrid candidate ranker for the Senior AI Engineer role at Redrob AI. Combines BM25 semantic matching over rich profile text with behavioral signals from Redrob, JD-specific feature overlap, career trajectory analysis, and experience-band fitting. Outputs ranked results in both CSV and XLSX.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python -m src.rank --data ./data/candidates.jsonl --out ./output/submission.csv
```

This produces both `submission.csv` (required) and `submission.xlsx`.

For local testing:
```bash
python -m src.rank --data ./data/sample_candidates.json --out ./output/test.csv --topk 20
```

## Validate

```bash
python data/validate_submission.py output/submission.csv
```

## Sandbox

`sandbox/colab_sandbox.ipynb` — runs the ranker on the included sample data (top 20).
