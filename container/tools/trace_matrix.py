#!/usr/bin/env python3
"""Build the requirement trace matrix from a pytest JUnit XML report.

A requirement is VERIFIED only if every test carrying its ID passed, OPEN if any
test for it is skipped with a PENDING reason, FAILED if any failed, and UNTESTED
if no test carries its ID at all.
"""
import re, sys
import xml.etree.ElementTree as ET
from collections import defaultdict

REQS = ["M1-01", "M1-02", "M1-03", "M1-04", "M1-05", "M1-06", "M2-01", "M2-02", "M2-03",
        "M3-01", "M3-02", "M3-03", "M3-04", "M3-05", "M3-06", "M3-07", "M3-08", "M3-09", "M3-10",
        "M3-11", "M3-12", "M3-13", "M12-01", "M12-02", "M12-03", "M12-04", "M12-05",
        "M13-01", "M13-02", "M13-03", "M13-04", "M13-05", "M13-06", "M13-07", "M13-08", "M13-09"]

root = ET.parse(sys.argv[1]).getroot()
res = defaultdict(list)
for tc in root.iter("testcase"):
    name = tc.get("name")
    ids = set()
    m = re.match(r"test_(M\d+)_(\d\d)((?:_\d\d)*)_", name)
    if m:
        ids.add(f"{m.group(1)}-{m.group(2)}")
        for extra in re.findall(r"_(\d\d)", m.group(3)):
            ids.add(f"{m.group(1)}-{extra}")
    p = re.search(r"\[(M\d+)_(\d\d)\]", name)
    if p:
        ids = {f"{p.group(1)}-{p.group(2)}"}
    if tc.find("failure") is not None or tc.find("error") is not None:
        st, why = "fail", (tc.find("failure") if tc.find("failure") is not None else tc.find("error")).get("message", "")[:120]
    elif tc.find("skipped") is not None:
        st, why = "skip", tc.find("skipped").get("message", "")
    else:
        st, why = "pass", ""
    for i in ids:
        res[i].append((name, st, why))

print("| Requirement | Status | Tests | Note |\n|---|---|---|---|")
counts = defaultdict(int)
for r in REQS:
    t = res.get(r, [])
    if not t:
        s, note = "UNTESTED", ""
    elif any(x[1] == "fail" for x in t):
        s, note = "FAILED", next(x[2] for x in t if x[1] == "fail")
    elif any(x[1] == "skip" for x in t):
        s, note = "OPEN", next(x[2] for x in t if x[1] == "skip").replace("Skipped: ", "")
    else:
        s, note = "VERIFIED", ""
    counts[s] += 1
    print(f"| {r} | **{s}** | {len(t)} | {note} |")
print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
