# India Runs Track 1

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python -m src.rank --data ./data/candidates.jsonl --out ./output/submission.csv
```

For local testing:
```bash
python -m src.rank --data ./data/sample_candidates.json --out ./output/test.csv --topk 20
```

## Validate

```bash
python data/validate_submission.py output/submission.csv
```

## Sandbox

Use `sandbox/colab_sandbox.ipynb` (share "Anyone with the link").