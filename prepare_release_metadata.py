"""Set actual GitHub URL/release date BEFORE creating the initial public tag."""
import argparse, datetime, hashlib, json, re
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-url", required=True)
    p.add_argument("--release-date", required=True, help="Actual planned ISO date YYYY-MM-DD")
    a=p.parse_args()
    url=a.repo_url.rstrip("/")
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",url):
        raise SystemExit("Use a real https://github.com/OWNER/REPOSITORY URL, without tokens.")
    datetime.date.fromisoformat(a.release_date)
    root=Path(__file__).resolve().parent
    cff=root/"CITATION.cff"
    s=cff.read_text(encoding="utf-8")
    s=re.sub(r"(?m)^(repository-code|date-released):.*\n?", "", s)
    cff.write_text(s+f'repository-code: "{url}"\ndate-released: "{a.release_date}"\n',encoding="utf-8")
    meta=json.loads((root/".zenodo.json").read_text())
    meta["publication_date"]=a.release_date
    meta["related_identifiers"]=[{"identifier":url,"relation":"isSupplementTo","scheme":"url"}]
    (root/".zenodo.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.name=="SHA256SUMS.txt" or any(x in f.parts for x in (".git","__pycache__")):
            continue
        rows.append(hashlib.sha256(f.read_bytes()).hexdigest()+"  "+f.relative_to(root).as_posix())
    (root/"SHA256SUMS.txt").write_text("\n".join(rows)+"\n")
    print("Metadata and code checksums updated. Inspect, commit, connect Zenodo, then publish v1.0.0.")
if __name__=="__main__": main()
