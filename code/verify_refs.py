"""Pull authoritative metadata for each candidate reference from Crossref (DOI or title search) and the arXiv API."""
import os as _os
ROOT = _os.environ.get("FPTM_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
CODE = _os.path.dirname(_os.path.abspath(__file__))
import json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r: return r.read()

def fmt(m):
    au = ", ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author", []))
    y = (m.get("published-print") or m.get("published-online") or m.get("issued") or {}).get("date-parts", [[None]])[0][0]
    return {"doi": m.get("DOI"), "title": (m.get("title") or [""])[0], "authors": au, "container": (m.get("container-title") or [""])[0],
            "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page"), "year": y, "publisher": m.get("publisher"), "type": m.get("type")}

def crossref_doi(doi): return fmt(json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"])
def crossref_search(q, rows=3):
    return [fmt(x) for x in json.loads(get("https://api.crossref.org/works?rows=%d&query.bibliographic=%s" % (rows, urllib.parse.quote(q))))["message"]["items"]]

def arxiv(aid):
    x = ET.fromstring(get("https://export.arxiv.org/api/query?id_list=" + aid)); ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = x.find("a:entry", ns); g = lambda p: (e.find(p, ns).text if e.find(p, ns) is not None else None)
    return {"arxiv": aid, "title": " ".join(e.find("a:title", ns).text.split()), "authors": ", ".join(a.find("a:name", ns).text for a in e.findall("a:author", ns)),
            "published": g("a:published")[:10], "journal_ref": g("x:journal_ref"), "doi": g("x:doi")}

DOIS = {"moos2021dlla": "10.46586/tches.v2021.i3.552-598", "moradi2018chi2": "10.13154/tches.v2018.i1.209-237",
        "bronchain2019": "10.13154/tches.v2019.i2.318-345", "yap2023ttdcnn": "10.46586/tches.v2023.i2.24-53",
        "gao2024explainable": "10.1007/978-3-031-58734-4_8", "levi2026gainp": "10.1016/j.sysarc.2025.103632",
        "yap2024occpois": "10.1007/978-3-031-80311-6_1", "ensan2021scare": "10.1109/TVLSI.2021.3110744",
        "fips203": "10.6028/NIST.FIPS.203", "lei2025leax": "10.1007/978-3-032-00627-1_16", "huang2022rpdcov": "10.3389/fams.2021.779841"}
SEARCH = {"chowdhury2026mvld": "Multivariate Leakage Detection Chowdhury Oswald Transactions on Cryptographic Hardware 2026",
          "schneider2015": "Leakage Assessment Methodology Schneider Moradi CHES 2015",
          "valk2020kilroy": "Kilroy was here: The first step towards explainability of neural networks in profiled side-channel analysis",
          "benadjila2020ascad": "Deep learning for side-channel analysis and introduction to ASCAD database Benadjila Prouff",
          "yoshida2025kan": "Kolmogorov-Arnold networks side-channel analysis Yoshida Karayalcin Picek",
          "wang2024imc": "Side-channel attack analysis on in-memory computing architectures Wang IEEE Transactions on Emerging Topics in Computing",
          "yavits2015rap": "Resistive associative processor Yavits Kvatinsky Morad Ginosar IEEE Computer Architecture Letters",
          "szekely2007": "Measuring and testing dependence by correlation of distances Szekely Rizzo Bakirov",
          "kocher1999": "Differential Power Analysis Kocher Jaffe Jun CRYPTO 1999",
          "chari1999": "Towards Sound Approaches to Counteract Power-Analysis Attacks Chari Jutla Rao Rohatgi",
          "heinz2022mkm4": "First-Order Masked Kyber on ARM Cortex-M4 Heinz Kannwischer Land Poppelmann Schwabe Sprenkels",
          "kannwischer2019pqm4": "pqm4 Testing and Benchmarking NIST PQC on ARM Cortex-M4 Kannwischer",
          "granmo2018": "The Tsetlin Machine A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic"}
ARXIV = {"hnilov2025fptm": "2508.08350", "granmo2018": "1804.01508", "karayalcin2025": "2502.00384", "gammell2024": "2410.22425",
         "bostanci2024impact": "2404.11284", "wang2023powergan": "2304.11056", "jaiswal2026ids": "2604.03205", "jaiswal2026pdf": "2607.09290",
         "wang2024imc": "2209.02792", "mikulec2026adla": "2603.18647", "peeking2026": "2609.33597"}
out = {}
for k, d in DOIS.items():
    try: out.setdefault(k, {})["crossref_doi"] = crossref_doi(d)
    except Exception as e: out.setdefault(k, {})["crossref_doi"] = f"ERROR {e}"
    time.sleep(0.3)
for k, q in SEARCH.items():
    try: out.setdefault(k, {})["crossref_search"] = crossref_search(q)
    except Exception as e: out.setdefault(k, {})["crossref_search"] = f"ERROR {e}"
    time.sleep(0.3)
for k, a in ARXIV.items():
    try: out.setdefault(k, {})["arxiv"] = arxiv(a)
    except Exception as e: out.setdefault(k, {})["arxiv"] = f"ERROR {e}"
    time.sleep(3.1)
json.dump(out, open(f"{ROOT}/results/refs_raw.json", "w"), indent=1)
for k, v in out.items():
    print("==", k)
    for src, val in v.items():
        if isinstance(val, list):
            for i, x in enumerate(val[:2]): print(f"  {src}[{i}]: {x['year']} | {x['title'][:90]} | {x['authors'][:80]} | {x['container'][:60]} {x['volume']}({x['issue']}) {x['pages']} | {x['doi']}")
        elif isinstance(val, dict): print(f"  {src}: " + " | ".join(f"{a}={str(b)[:90]}" for a, b in val.items()))
        else: print(f"  {src}: {val}")
