
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

DATASET = Path(__file__).with_name("raw_skills.csv")
MIN_SKILLS = 3   # project requirement: at least three user inputs
TOP_N = 3

ALIASES = {
    "ml": "machine learning",
    "ai": "machine learning",
    "dl": "deep learning",
    "cloud computing": "cloud",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "node": "node.js",
    "nodejs": "node.js",
    "postgres": "postgresql",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "api": "apis",
    "rest api": "apis",
    "dsa": "data structures",
    "algorithms": "data structures",
    "network": "networking",
    "cyber security": "security",
    "cybersecurity": "security",
    "tf": "tensorflow",
    "dataviz": "data visualization",
    "tcpip": "tcp/ip",
    "csharp": "c#",
    "cpp": "c++",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_roles(path=DATASET):
    """Return {role: [skills...]} from the CSV (skills separated by ';')."""
    roles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            skills = [normalize(s) for s in row["skills"].split(";") if s.strip()]
            roles[row["role"].strip()] = skills
    return roles


def normalize(term):
    term = " ".join(term.lower().strip().split())
    return ALIASES.get(term, term)



def build_vocabulary(roles):
    return sorted({s for skills in roles.values() for s in skills})


def compute_idf(roles, vocab):
    """IDF = log(total documents / documents containing the term)."""
    n_docs = len(roles)
    idf = {}
    for term in vocab:
        df = sum(1 for skills in roles.values() if term in skills)
        idf[term] = math.log(n_docs / df)
    return idf


def tfidf_vector(terms, vocab, idf):
    """TF = count(term) / total terms; weight = TF * IDF, in vocab order."""
    total = len(terms)
    counts = Counter(terms)
    if total == 0:
        return [0.0] * len(vocab)
    return [(counts[t] / total) * idf[t] for t in vocab]


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #
def cosine_similarity(a, b):
    """cos(theta) = (A . B) / (||A|| * ||B||); 0.0 if either vector is zero."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)



def trending_fallback(roles, top_n=TOP_N):
    """Cold-start bypass: if no usable input, return the roles with the
    broadest skill sets as a 'global popularity' style default."""
    ranked = sorted(roles, key=lambda r: len(roles[r]), reverse=True)
    return [(r, 0.0, [], []) for r in ranked[:top_n]]


def recommend(user_skills, roles, top_n=TOP_N):
    """
    Steps:  1 Ingestion  2 Scoring  3 Sorting  4 Filtering
    Returns (results, unknown_skills) where results is a list of
    (role, score, matched_skills, missing_skills).
    """
    vocab = build_vocabulary(roles)
    idf = compute_idf(roles, vocab)

    
    cleaned = [normalize(s) for s in user_skills if s.strip()]
    known = [s for s in cleaned if s in vocab]
    unknown = [s for s in cleaned if s not in vocab]

    if not known:                      
        return trending_fallback(roles, top_n), unknown

    user_vec = tfidf_vector(known, vocab, idf)

    
    scored = []
    for role, skills in roles.items():
        role_vec = tfidf_vector(skills, vocab, idf)
        score = cosine_similarity(user_vec, role_vec)
        matched = sorted(set(known) & set(skills))
        missing = sorted(set(skills) - set(known))
        scored.append((role, score, matched, missing))

    scored.sort(key=lambda x: (-x[1], x[0]))


    scored = [row for row in scored if row[1] > 0]
    if not scored:
        return trending_fallback(roles, top_n), unknown
    return scored[:top_n], unknown


def parse_skills(raw):
    return [s for s in raw.split(",") if s.strip()]


def prompt_for_skills(vocab):
    print("=" * 60)
    print("  TECH STACK RECOMMENDER  |  DecodeLabs AI Project 3")
    print("=" * 60)
    print("Known skills include:\n  " + ", ".join(vocab) + "\n")
    while True:
        raw = input(f"Enter at least {MIN_SKILLS} skills (comma-separated): ")
        skills = parse_skills(raw)
        if len(skills) >= MIN_SKILLS:
            return skills
        print(f"  ! Please provide at least {MIN_SKILLS} skills.\n")


def display(results, unknown, fallback=False):
    print()
    if unknown:
        print(f"Ignored (not in dataset): {', '.join(unknown)}\n")
    if fallback:
        print("No recognised skills -- showing broad-skill-set roles instead.\n")
    print(f"TOP {len(results)} CAREER PATHS")
    print("-" * 60)
    for i, (role, score, matched, missing) in enumerate(results, 1):
        print(f"{i}. {role}  --  {score * 100:.1f}% match")
        if matched:
            print(f"   You have : {', '.join(matched)}")
        if missing:
            print(f"   To learn : {', '.join(missing)}")
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tech Stack Recommender")
    parser.add_argument("--skills", help='comma-separated, e.g. "python,cloud,automation"')
    parser.add_argument("--top", type=int, default=TOP_N, help="number of results")
    parser.add_argument("--data", default=str(DATASET), help="path to skills CSV")
    args = parser.parse_args(argv)

    roles = load_roles(args.data)
    if args.skills:
        skills = parse_skills(args.skills)
        if len(skills) < MIN_SKILLS:
            sys.exit(f"Error: provide at least {MIN_SKILLS} skills.")
    else:
        skills = prompt_for_skills(build_vocabulary(roles))

    results, unknown = recommend(skills, roles, args.top)
    fallback = all(score == 0.0 for _, score, _, _ in results)
    display(results, unknown, fallback)


if __name__ == "__main__":
    main()
