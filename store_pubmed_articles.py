import requests
import json
import time
from xml.etree import ElementTree as ET
from Bio import Entrez

# ===============================
# CONFIG
# ===============================
Entrez.email = "email@email.com.br"

OUTPUT_FILE = "articles_with_fulltext5.json"

PMIDS = [
    "32528504",
    "20584286",
    "25148240",
    "24204619",
    "28117379",
    #"26819184",
    #"31711424",
    #"25209012", 
    #"31711424", 
    #"32275650"
]

UA = {"User-Agent": f"pmc-scraper/1.0 ({Entrez.email})"}


# ===============================
# PMID → PMCID
# ===============================
def pmid_to_pmcid(pmid):
    # Method 1: NCBI ID Converter
    try:
        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json",
            timeout=10
        ).json()
        rec = r["records"][0]
        if "pmcid" in rec:
            return rec["pmcid"]
    except:
        pass

    # Method 2: Europe PMC
    try:
        r = requests.get(
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}&format=json",
            timeout=10
        ).json()
        results = r.get("resultList", {}).get("result", [])
        for item in results:
            if "pmcid" in item:
                return item["pmcid"]
    except:
        pass

    return None


# ===============================
# Check PMC Open Access status
# ===============================
def check_oa(pmcid):
    try:
        r = requests.get(
            f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}",
            timeout=10
        )
        root = ET.fromstring(r.content)

        record = root.find(".//record")
        if record is None:
            return False

        links = record.findall(".//link")
        if not links:
            return False

        return True
    except:
        return False


# ===============================
# Fetch PubMed metadata
# ===============================
def fetch_metadata(pmid):
    try:
        h = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
        data = Entrez.read(h)
        h.close()

        article = data["PubmedArticle"][0]["MedlineCitation"]["Article"]

        title = article.get("ArticleTitle", "")
        journal = article["Journal"]["Title"]
        year = article["Journal"]["JournalIssue"]["PubDate"].get("Year", "")

        authors = []
        for a in article.get("AuthorList", []):
            try:
                authors.append(f"{a['LastName']}, {a['ForeName']}")
            except:
                pass

        abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else ""

        return {
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
            "authors": authors,
            "abstract": abstract
        }
    except:
        return None


# ===============================
# Fetch JATS full text
# ===============================
def fetch_fulltext(pmcid):
    try:
        num = pmcid.replace("PMC", "")

        url = (
            "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
            f"?verb=GetRecord&metadataPrefix=pmc&identifier=oai:pubmedcentral.nih.gov:{num}"
        )
        r = requests.get(url, headers=UA, timeout=15)
        root = ET.fromstring(r.content)

        # Look for <article>
        article = root.find(".//{*}article")
        if article is None:
            return None

        text_blocks = []

        # paragraphs
        for p in article.findall(".//{*}body//{*}p"):
            txt = "".join(p.itertext()).strip()
            if txt:
                text_blocks.append(txt)

        return "\n\n".join(text_blocks) if text_blocks else None

    except:
        return None


# ===============================
# MAIN PIPELINE
# ===============================

stored_articles = []

for pmid in PMIDS:
    print(f"\nProcessing PMID {pmid}...")

    metadata = fetch_metadata(pmid)
    if metadata is None:
        print("Could not fetch metadata.")
        continue

    pmcid = pmid_to_pmcid(pmid)
    metadata["pmcid"] = pmcid

    if pmcid is None:
        print("No PMCID found → skipping.")
        continue

    print(f"PMCID: {pmcid}")

    # Check OA subset
    if not check_oa(pmcid):
        print("Not in OA subset → full text legally inaccessible → skipping.")
        continue

    print("OA subset confirmed.")

    # Try full text
    fulltext = fetch_fulltext(pmcid)
    if not fulltext:
        print("Full text could not be retrieved → skipping.")
        continue

    print("Full text retrieved.")

    metadata["full_text"] = fulltext
    stored_articles.append(metadata)

    time.sleep(1)

# Save results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(stored_articles, f, ensure_ascii=False, indent=2)

print("\n==============================")
print("DONE!")
print(f"Stored articles with full text: {len(stored_articles)}")
print(f"Saved to: {OUTPUT_FILE}")
print("==============================")
