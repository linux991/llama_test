# Horror Corpus Preparation Report

## Corpus

- Stories: 3446
- Train stories: 2756
- Eval stories: 690
- Story chars: min=504, median=10141, mean=12553, p95=32767, max=32767
- Story words: min=86, median=1918, mean=2380, p95=6154, max=6703

## Chunks

- Full train chunks: 24201
- Full eval chunks: 5787
- Removed eval chunks due to exact train overlap: 18
- Payload train chunks: 2000
- Payload eval chunks: 500
- Chunk chars: min=700, median=1454, mean=1421, p95=1496, max=1500

## Leakage Checks

- Overlapping story hashes between train/eval: 0
- Overlapping chunk hashes between train/eval: 0

## Colab Payload

- Zip: `data/horror_experiment_payload.zip`

Payload files:

- `train_with_descriptions.csv`
- `eval_with_descriptions.csv`
- `split_metadata.csv`
- `corpus_summary.csv`
- `corpus_preparation_report.md`
