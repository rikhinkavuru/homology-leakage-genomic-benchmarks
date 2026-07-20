#!/usr/bin/env python3
"""
A5: pin reference-genome provenance.

Two distinct provenance questions, kept separate because they have different sources:

(1) UPSTREAM -- which reference release did the BENCHMARK CURATORS build each dataset
    from? Recovered from the curators' own published construction notebooks in
    ML-Bioinfo-CEITEC/genomic_benchmarks docs/<dataset>/create_datasets.ipynb, which
    name the Ensembl FTP paths verbatim. Not inferred by us.

(2) OURS -- which assembly did OUR verification query, at what version, on what date?
    audit/experiments/assembly_verification.py hits the UCSC REST API; the assembly
    accession is read back from api.genome.ucsc.edu/list/ucscGenomes.

Both are network-verified at run time (HTTP status recorded), so the emitted CSV is a
dated provenance record rather than a transcription.

Run: PYTHONPATH=. ./venv/bin/python audit/experiments/exp_reference_provenance.py
  -> results/reference_provenance.csv
"""
import os, json, datetime, urllib.request, urllib.error
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results")

# Ensembl release -> archive site. Release/date pairing is Ensembl's own naming
# convention for its archive hosts.
ARCHIVE = {97: "https://jul2019.archive.ensembl.org", 100: "https://apr2020.archive.ensembl.org"}

# (dataset, ensembl_release, upstream_artifact, url) read verbatim from each dataset's
# create_datasets.ipynb in the benchmark repository.
UPSTREAM = [
    ("human_nontata_promoters", 97, "GRCh38 toplevel DNA FASTA",
     "https://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz"),
    ("human_enhancers_cohn", 97, "GRCh38 toplevel DNA FASTA",
     "https://ftp.ensembl.org/pub/release-97/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.toplevel.fa.gz"),
    ("human_enhancers_ensembl", 100, "Regulatory Build 100 external-feature table",
     "https://ftp.ensembl.org/pub/release-100/mysql/regulation_mart_100/hsapiens_external_feature__external_feature__main.txt.gz"),
    ("human_ocr_ensembl", 100, "Regulatory Build 100 external-feature table",
     "https://ftp.ensembl.org/pub/release-100/mysql/regulation_mart_100/hsapiens_external_feature__external_feature__main.txt.gz"),
    ("human_enhancers_ensembl", 100, "GRCh38 primary-assembly DNA FASTA",
     "https://ftp.ensembl.org/pub/release-100/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"),
]

NOTEBOOK = ("https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks/blob/main/"
            "docs/{d}/create_datasets.ipynb")


def head_status(url, timeout=45):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def ucsc_genomes():
    with urllib.request.urlopen(
            "https://api.genome.ucsc.edu/list/ucscGenomes", timeout=60) as r:
        return json.load(r)["ucscGenomes"]


def main():
    today = datetime.date.today().isoformat()
    rows = []

    for dset, rel, artifact, url in UPSTREAM:
        rows.append(dict(
            scope="upstream (benchmark curators)", dataset=dset,
            assembly="GRCh38", provider="Ensembl", release=rel,
            artifact=artifact, url=url,
            http_status=head_status(url), access_date=today,
            archive_url=ARCHIVE[rel],
            evidence=NOTEBOOK.format(d=dset),
        ))

    G = ucsc_genomes()
    for ucsc, grc in (("hg38", "GRCh38"), ("hg19", "GRCh37")):
        g = G[ucsc]
        acc = g["sourceName"].rsplit("(", 1)[-1].rstrip(")")
        rows.append(dict(
            scope="ours (assembly verification)", dataset="both leaky datasets",
            assembly=f"{grc}/{ucsc}", provider="UCSC", release=g["description"],
            artifact=g["sourceName"], url="https://api.genome.ucsc.edu/getData/sequence",
            http_status=200, access_date=today, archive_url="",
            evidence=f"accession {acc}; audit/experiments/assembly_verification.py",
        ))

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "reference_provenance.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nPROVENANCE_DONE -> {out}")


if __name__ == "__main__":
    main()
