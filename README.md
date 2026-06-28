# India Runs Track 1

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

`sandbox/colab_sandbox.ipynb`
