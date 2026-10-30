# Tech Stack Recommender

> **DecodeLabs · Artificial Intelligence · Project 3: AI Recommendation Logic**

A content-based recommendation engine that maps a user's skills to the most relevant tech career paths. It is built **from scratch in pure Python** (standard library only) using **TF-IDF weighting** and **cosine similarity**, with no scikit-learn or NumPy.

## How it works

The engine follows the Input → Process → Output model and a 4-step ranking pipeline:

| Step | Name | What happens |
|------|------|--------------|
| 1 | **Ingestion** | Takes at least 3 skills from the user and normalises them (`"Cloud Computing"` → `cloud`, `"k8s"` → `kubernetes`) so the user and the dataset share one vocabulary |
| 2 | **Scoring** | Builds TF-IDF vectors for the user and every job role, then computes cosine similarity |
| 3 | **Sorting** | Ranks roles by score, descending |
| 4 | **Filtering** | Drops zero-score roles and returns the Top-N (default 3) |

### The maths

```
TF(t, d)  = count of t in d / total terms in d
IDF(t)    = log(total documents / documents containing t)
weight    = TF × IDF

cos(θ)    = (A · B) / (‖A‖ × ‖B‖)
```

- **IDF** penalises generic skills (e.g. `python`, `git`) and rewards specific ones (e.g. `tensorflow`, `terraform`).
- **Cosine similarity** compares the *direction* of the vectors, not their size. Since TF-IDF values are non-negative, scores fall in `[0, 1]` and read naturally as a percentage match.

### Cold-start handling

- **Item cold start:** new roles can be added to the CSV and are recommended immediately, based on their skills.
- **User cold start:** if none of the entered skills are recognised, the script falls back to the roles with the broadest skill sets (a "trending" style default) instead of crashing or returning all zeros. Unknown skills are listed so the user can correct them.

## Project structure

```
tech-stack-recommender/
├── recommender.py          # the engine + CLI
├── raw_skills.csv          # dataset: role, skills (semicolon-separated)
├── tests/
│   └── test_recommender.py # unit tests
└── README.md
```

## Getting started

**Requirements:** Python 3.8+ (no third-party packages).

```bash
git clone https://github.com/<your-username>/tech-stack-recommender.git
cd tech-stack-recommender
```

### Interactive mode

```bash
python recommender.py
```

### One-line mode

```bash
python recommender.py --skills "Python,Cloud Computing,Automation"
```

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `--skills` | Comma-separated skills (minimum 3) | prompts interactively |
| `--top` | Number of recommendations | `3` |
| `--data` | Path to a custom skills CSV | `raw_skills.csv` |

## Example output

```
$ python recommender.py --skills "Python,Cloud Computing,Automation"

TOP 3 CAREER PATHS
------------------------------------------------------------
1. Cloud Engineer  --  57.7% match
   You have : automation, cloud, python
   To learn : aws, azure, docker, linux, networking, terraform

2. Cloud Architect  --  50.3% match
   You have : automation, cloud
   To learn : aws, azure, kubernetes, linux, networking, security, terraform

3. DevOps Engineer  --  11.0% match
   You have : automation, python
   To learn : aws, ci/cd, docker, git, jenkins, kubernetes, linux, terraform
```

Besides the ranking, each result shows the skills you already have and a **skill gap** of what to learn next.

## Dataset format

`raw_skills.csv` treats each job role as an "item":

```csv
role,skills
DevOps Engineer,aws;docker;kubernetes;ci/cd;git;linux;terraform;automation;python;jenkins
```

To add a role, append a row. Use lowercase skill names, and add any alternative spellings to the `ALIASES` dictionary in `recommender.py`.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The tests cover cosine similarity edge cases (identical, orthogonal and zero vectors), IDF weighting, alias normalisation, ranking order, unknown-skill handling and the cold-start fallback.

## Known limitations and ideas

- The vocabulary is limited to the skills in the CSV, so unseen skills are ignored (and reported).
- With a small dataset, IDF values are coarse. A larger dataset would give better weighting.
- Possible extensions: skill ratings (1–5) as weights, a web UI (Flask/Streamlit), and a hybrid model with collaborative filtering.

## Concepts demonstrated

Content-based filtering · feature extraction · vector mapping · TF-IDF · cosine similarity · top-N ranking · cold-start mitigation

## License

MIT
