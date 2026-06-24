"""Decode the Transdiagnostic tmb_dsm_score values - find SZ/BD/MDD mapping."""
import csv
from collections import Counter
from pathlib import Path

TD = Path(r"F:\PQFL_Schizophrenia_Full_Project\pqfl_project\data\Transdiagnostic")

# 1. Read the definitions file
defs_path = TD / "phenotype" / "tmb_dsm01_definitions.tsv"
print("=" * 78)
print("[1] tmb_dsm01_definitions.tsv - FULL CONTENT")
print("=" * 78)
with open(defs_path, encoding="utf-8") as f:
    for line in f:
        print(line.rstrip())

# 2. Read the data file and show distribution
print()
print("=" * 78)
print("[2] tmb_dsm01.tsv - tmb_dsm_score distribution")
print("=" * 78)
data_path = TD / "phenotype" / "tmb_dsm01.tsv"
with open(data_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)
print(f"Total subjects: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")
scores = [r["tmb_dsm_score"].strip() for r in rows if r.get("tmb_dsm_score")]
score_counter = Counter(scores)
print(f"\ntmb_dsm_score value distribution:")
for score, n in sorted(score_counter.items(), key=lambda x: -x[1]):
    print(f"  {score:<10}  {n:>4}")

# 3. Cross-reference with participants.tsv Group column
print()
print("=" * 78)
print("[3] Cross-tab: tmb_dsm_score x participants.tsv Group")
print("=" * 78)
participants_path = TD / "participants.tsv"
with open(participants_path, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    participants = list(reader)

# Build subjectkey -> Group mapping
# participant_id like "sub-NDARINVBB020WYD" -> subjectkey like "NDAR_INVBB020WYD"
sk_to_group = {}
sk_to_site = {}
for r in participants:
    pid = r["participant_id"].strip()
    group = r.get("Group", "").strip()
    site = r.get("Site", "").strip()
    if pid.startswith("sub-NDARINV"):
        sk = pid.replace("sub-NDARINV", "NDAR_INV")
        sk_to_group[sk] = group
        sk_to_site[sk] = site

# Build cross-tab
cross = {}
for r in rows:
    sk = r.get("subjectkey", "").strip()
    score = r.get("tmb_dsm_score", "").strip()
    group = sk_to_group.get(sk, "UNKNOWN")
    if group not in cross:
        cross[group] = Counter()
    cross[group][score] += 1

print(f"\n{'Group':<15}  Score distribution")
print("-" * 78)
for group in sorted(cross):
    scores_str = ", ".join(f"{s}={n}" for s, n in sorted(cross[group].items()))
    total = sum(cross[group].values())
    print(f"{group:<15}  (n={total}): {scores_str}")

# 4. Also show Site x Group x Score
print()
print("=" * 78)
print("[4] Site x Group x Score (full breakdown)")
print("=" * 78)
triple = {}
for r in rows:
    sk = r.get("subjectkey", "").strip()
    score = r.get("tmb_dsm_score", "").strip()
    group = sk_to_group.get(sk, "UNKNOWN")
    site = sk_to_site.get(sk, "UNKNOWN")
    key = (site, group)
    if key not in triple:
        triple[key] = Counter()
    triple[key][score] += 1

print(f"\n{'Site':<6}  {'Group':<10}  Score distribution")
print("-" * 78)
for (site, group) in sorted(triple.items()):
    scores_str = ", ".join(f"{s}={n}" for s, n in sorted(triple[(site, group)].items()))
    total = sum(triple[(site, group)].values())
    print(f"{site:<6}  {group:<10}  (n={total}): {scores_str}")

# 5. Try to interpret
print()
print("=" * 78)
print("[5] INTERPRETATION GUIDE")
print("=" * 78)
print("  Based on TCP/Transdiagnostic Connectome Project documentation,")
print("  the tmb_dsm_score is likely a NDA enum where:")
print("    1  = Healthy Control (no DSM diagnosis)")
print("    2  = Schizophrenia Spectrum (DSM 295.x)")
print("    3  = Bipolar Disorder (DSM 296.x)")
print("    4  = Major Depressive Disorder (DSM 296.2x/296.3x)")
print("    5  = Anxiety Disorder (DSM 300.x)")
print("    6  = ADHD (DSM 314.x)")
print("    7+ = Other conditions")
print()
print("  GenPop subjects should mostly have score=1 (HC)")
print("  Patient subjects should have varied scores (2-7)")
print("  The cross-tab in [3] will reveal the actual mapping.")
print("=" * 78)
