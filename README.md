# SimulacraBench submission

This is the code for team **Poverty Killer**'s Test-phase submission to
[SimulacraBench](https://simulacrabench.org/). SimulacraBench is a NeurIPS 2026
competition on predicting held-out survey answers in a UNHCR, a UNICEF and a World Bank
instrument. The technical report, [`technical_report.pdf`](technical_report.pdf), describes
the method; it is the write-up submitted with the entry on OpenReview.

## Contents

| Path | What it is |
|---|---|
| `technical_report.pdf` | The technical report: method, Test-phase results, limitations and full configuration. |
| `submission/main.py` | `predict(frame, schema)`, exactly as uploaded to the Test phase (`BUDGET = 3600`). |
| `submission/requirements.txt` | Packages installed into the hosted image: `tabpfn` and its pinned dependencies. |
| `submission/models.txt` | The Hugging Face model the platform downloads before the clock starts: `Prior-Labs/TabPFN-v2-clf`. |
| `scripts/make_zip.py` | Builds the upload archive with sorted entries and a fixed timestamp, so the same files always give the same SHA-256. |

## Method in brief

Everything is fit inside the timed run, on the respondents whose answers are visible.

**Candidate models.** For every question to predict, `main.py` fits several probabilistic models on the visible respondents' known answers:
- multinomial logistic regression, plus proportional-odds logistic regression for ordered items;
- gradient-boosted trees;
- country tables shrunk toward pooled shares;
- a multi-task network over all of an instrument's items;
- TabPFN v2 on the GPU.

**Blending.** The candidates are blended. The weights, the pooling and a temperature are chosen on held-out visible respondents.

**Skip logic.** Skip logic is modelled as gate chains, with "asked" rates learned for each answer of the parent question, because many declared gates hold only approximately. For each gated item, a direct model of its full distribution competes with the chain.

**Averaging.** In the Test phase the whole pipeline runs up to five times per instrument, each time on a different held-out split. The resulting probability vectors are averaged.

The docstring of `main.py` records what changed in every version.

## Reproduce the uploaded archive

```bash
python scripts/make_zip.py submission test-v13.zip
```

The result is byte-identical to the archive uploaded to the Test phase. Its SHA-256 is
`76df4bb8556f9dc107d74f2a0c017eb7a8bfab20a93d6f84d8dcc489cbfcea17`.

## Run it locally

The survey microdata never leave the organizers' hosted runtime. Instead, the official
kit builds sandbox data that uses the real schemas with invented answers. It then scores
a submission in the Test-phase setting: `TRAIN` and `DEV` respondents are visible, and
`TEST` respondents are scored.

```bash
git clone https://github.com/SituatedEvals/public SituatedEvals-public
cd SituatedEvals-public
pip install -r requirements.txt
for s in unhcr unicef world_bank; do
  python make_sandbox.py --schema data/$s.json --out _sandbox/$s
  python score.py --submission ../simulacrabench-submission/submission \
      --data _sandbox/$s --schema data/$s.json --phase 2
done
```

Because the sandbox answers are invented, sandbox scores show only that the code runs in
time and returns valid vectors. They say nothing about skill on the real data. Without a
GPU or the TabPFN checkpoint, TabPFN is skipped and the rest of the blend runs unchanged.

## Environment

The organizers describe the hosted runtime as follows:
- one H100 GPU, 8 CPU cores and 16 GB of RAM, with no network;
- 3600 s in the Test phase for all three instruments together;
- packages in `requirements.txt` are installed when the image is built;
- the model in `models.txt` is downloaded before the clock starts.

Apart from `tabpfn` and `torch`, `main.py` uses numpy, pandas, scipy, scikit-learn and
joblib from the organizers' image.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The
TabPFN v2 weights are not part of this repository. The platform downloads them from
Hugging Face under Prior Labs' license.
