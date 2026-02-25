import argparse
import ast
import html
import json
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import pandas as pd
from molecule.llm_judge.code.graph_utils import (
    build_lookup_dicts,
    compute_graph_stats,
    get_canon_smiles,
    get_tags,
)

# from mu0_rag import get
from openai import OpenAI

# from distutils.util import strtobool
from setuptools._distutils.util import strtobool
from tqdm import tqdm
from molecule.llm_judge.code.utils import load_json, save_json



def get_file_dirs(jsonl_path, meta_csv_path):
    directories = {
        jsonl_path: meta_csv_path
        ### MD image folder also need to be added in graph_utils.py
        }
    return directories


def replace(s, stop_strings, replacement=""):
    for ss in stop_strings:
        s = s.replace(ss, replacement)
    return s


def build_file_info_validated(output_path, file_info_validated, jsonl_path, meta_csv_path) -> List[str]:
    def build_paper_citation(doc_id, row):
        
        # Title; Escape title for safe HTML
        title = "Title unknown."
        for title_str in ["title", "Title", "result_title"]: # result_title for patent csv
            if title_str in row and pd.notna(row[title_str]):
                title = html.escape(row[title_str])
                break

        # Hyperlink
        hyperlinked_title = "No hyperlink available."
        if "result_publication_number" in row and pd.notna(row["result_pdf_link"]): # meaning it's from a patent table
            hyperlinked_title = (
                f'<a href="{row["result_pdf_link"]}" target="_blank" rel="noopener noreferrer">'
                f"{title}</a>"
            )
        else:
            doi_url = f"https://doi.org/{doc_id}"
            hyperlinked_title = (
                f'<a href="{doi_url}" target="_blank" rel="noopener noreferrer">'
                f"{title}</a>"
            )

        # Format authors as a natural list (commas and 'and')
        authors_formatted = "Authors unknown."
        for author_str in ["authors", "author", "Senior Corresponding Author", "result_inventor"]: # result_inventor for patent
            if author_str in row and pd.notna(row[author_str]):
                authors = row[author_str]
                author_list = [a.strip() for a in authors.split(";")]
                if len(author_list) == 1:
                    authors_formatted = author_list[0]
                elif len(author_list) == 2:
                    authors_formatted = f"{author_list[0]} and {author_list[1]}"
                else:
                    authors_formatted = (
                        ", ".join(author_list[:-1]) + f", and {author_list[-1]}"
                    )
                break
                
        # Journal
        journal = "Journal unknown"
        for j_str in ["journal", "publication", "Journal", "result_assignee_field"]: # result_assignee_field for patent
            if j_str in row and pd.notna(row[j_str]):
                journal = row[j_str]
                break

        # Publication Year
        year = "Year unknown"
        for y_str in ["year", "publication_year", "Year", "result_publication_date"]: # result_publication_date for year
            if y_str in row and pd.notna(row[y_str]):
                if y_str == "result_publication_date":
                    year = row[y_str][:4]
                else:
                    year = row[y_str]
                break

        # Build ACS‑style citation (string containing HTML markup)
        citation = (
            f"{authors_formatted}. {hyperlinked_title}. "
            f"{journal}, {year}. [DOI: {doc_id}]"
        )
        return citation

    def build_citation(doc_id, row=None, path=None):
        if path is None and row is None:
            raise Exception(
                f"Cannot build a citation without metadata. (doc_id:{doc_id})"
            )
        if row is not None:
            # Metadata available
            citation = build_paper_citation(doc_id=doc_id, row=row)
        else:
            # Metadata extracted from filename
            citation = (
                Path(path)
                .stem.replace("-0", "")
                .replace("_", " ")
                .replace("(Z-Lib.io)", "")
                .replace("libgen.li", "")
                .replace("-", " ")
                .replace("+", " ")
                .title()
                .replace("(Z Lib.Io)", "")
                .replace("Libgen.Lc", "")
                .strip()
            ) + f". [DOC-ID: {doc_id}]"

        return re.sub(r"\s{2,}", " ", citation)

    def build_path(doc_id, directory):
        prefix = doc_id.split("/")[0].strip() + "_" if "3p6_jsonl" in directory else ""
        filename = prefix + doc_id.split("/")[-1].strip()
        path = directory + filename + "-0.jsonl"
        return path

    def fix_doi(doi):
        return replace(
            doi,
            [
                "htps://doi.org/",
                "/doi.org",
                "doi.org/",
                "\u00f4\u00aa\u00f8",
            ],
        )

    directories = get_file_dirs(jsonl_path, meta_csv_path)
    for directory, excel_file in directories.items():
        if excel_file:
            try:
                df = pd.read_excel(excel_file) if Path(excel_file).suffix in ['.xls', '.xlsx'] else pd.read_csv(excel_file)
                if "doi" in df.columns:
                    df = df.dropna(subset=["doi"]).reset_index(drop=True)
                if "DOI" in df.columns:
                    df = df.dropna(subset=["DOI"]).reset_index(drop=True)
                if "result_publication_number" in df.columns:
                    df = df.dropna(subset=["result_publication_number"]).reset_index(drop=True)
                if "path" in df.columns:
                    excel_paths = df["path"].to_list()
                    excel_paths = [
                        p if isinstance(p, str) else None for p in excel_paths
                    ]
                else:
                    excel_paths = None
                for i in range(df.shape[0]):
                    if "doi" in df.columns:
                        doc_id = str(df["doi"][i]).lower().strip()
                        doc_id = fix_doi(doc_id)
                    elif "DOI" in df.columns:
                        doc_id = str(df["DOI"][i]).lower().strip()
                        doc_id = fix_doi(doc_id)
                    else:
                        doc_id = str(df["result_publication_number"][i]).strip()
                        
                    if (doc_id in file_info_validated) and (
                        file_info_validated[doc_id].get("path", None)
                    ):
                        # This doc_id has already been fully processed
                        assert (
                            file_info_validated[doc_id]["citation"] != ""
                        ), f"Citation for {doc_id} is missing."
                        continue
                    else:
                        # This doc_id has NOT been processed yet
                        if doc_id not in file_info_validated:
                            file_info_validated[doc_id] = {}
                            file_info_validated[doc_id]["citation"] = build_citation(
                                doc_id=doc_id,
                                row=df.iloc[i],
                            )
                            file_info_validated[doc_id]["paths"] = []
                            if excel_paths and excel_paths[i]:
                                path = excel_paths[i]
                                file_info_validated[doc_id]["path"] = path
                            else:
                                test_path = build_path(doc_id, directory)
                                if (
                                    Path(test_path).is_file()
                                    and read_file(test_path) != ""
                                ):
                                    file_info_validated[doc_id]["path"] = test_path
                                else:
                                    file_info_validated[doc_id]["paths"].append(test_path)
                        else:
                            # This doc_id has already been seen but a final path has not been found yet
                            file_info_validated[doc_id]["paths"].append(
                                build_path(doc_id, directory)
                            )
            except Exception as e:
                print(f"Error reading Excel file or generating paths: {e}")
        else:  # We assume these are books without a valid doc_id and create an id called book_id_i
            files = [str(f) for f in Path(directory).glob("*.jsonl")]
            for ifile, path in enumerate(files):
                book_id = f"book_{ifile}"
                if book_id not in file_info_validated:
                    file_info_validated[book_id] = {}
                    file_info_validated[book_id]["path"] = path
                    file_info_validated[book_id]["citation"] = build_citation(
                        doc_id=book_id,
                        path=path,
                    )

    # Find valid paths for doc_ids without 'path' field
    for doc_id, doc_id_dict in file_info_validated.items():
        if not doc_id_dict.get("path"):
            file_info_validated[doc_id]["path"] = find_path(
                doc_id=doc_id,
                paths=file_info_validated[doc_id]["paths"],
                jsonl_path=jsonl_path,
                meta_csv_path=meta_csv_path,
            )

    # Remove files that we will not be able to read
    doc_ids_to_remove = []
    for doc_id, doc_id_dict in file_info_validated.items():
        file_info_validated[doc_id].pop("paths", None)
        if ("path" not in doc_id_dict) or (read_file(doc_id_dict["path"]) in (None, "")):
            doc_ids_to_remove.append(doc_id)
    for doc_id in doc_ids_to_remove:
        file_info_validated.pop(doc_id)
    return file_info_validated


def find_path(doc_id, paths, jsonl_path, meta_csv_path):
    all_paths = get_all_paths(jsonl_path, meta_csv_path)
    path = get_valid_option(paths, doc_id, all_paths)
    return path



def get_all_paths(jsonl_path, meta_csv_path):
    dirs = get_file_dirs(jsonl_path, meta_csv_path)
    files = []
    for d, e in dirs.items():
        if "3p6_jsonl" not in d and "textbooks" not in d:
            files.extend([str(p) for p in Path(d).rglob("*.jsonl")])
    return files


def find_misplaced(
    all_paths,
    doc_id,
):
    doc_id_last = re.split(r"[^a-zA-Z0-9]+", doc_id)[-1]
    if len(doc_id_last) > 5:
        for p in all_paths:
            if doc_id_last in p:
                print(f"Found misplaced file with doc_id: {doc_id} in {p}")
                return p
    else:
        return None


def get_valid_option(options, doc_id, all_paths):
    variants = []
    for option in options:
        parent, filename = option.rsplit("/", 1)
        variants.extend(
            [
                option,
                parent + "/" + filename.lower(),
                parent + "/" + filename.upper(),
            ]
        )
    good_variant = [
        variant
        for variant in variants
        if Path(variant).is_file() and read_file(variant) != ""
    ]
    if len(good_variant) == 0:
        good_variant = find_misplaced(all_paths, doc_id)
        return good_variant
    else:
        return good_variant[0]


def read_file(file_path):
    print(f"Reading file {file_path}")
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            line = next(f)
            doc = json.loads(line.strip())
            return doc.get("text", "")
    else:
        return ""


def read_jsonl_files(
    file_info_validated: dict[str, str],
) -> List[Dict]:

    documents = {}
    # Read document text
    for doc_id, doc_id_dict in tqdm(file_info_validated.items(), desc="Reading files"):
        text = None
        try:
            text = read_file(doc_id_dict["path"])
        except:
            print(f"Could not read file {doc_id_dict['path']} for doc_id {doc_id}")
        if text and text != "":
            documents[doc_id] = text
    print(f"Successfully read {len(documents)}/{len(file_info_validated)} files")
    return documents



def manage_res_fld(output_path: str):
    os.makedirs(output_path, exist_ok=True)
    return


def get_documents(output_path, jsonl_path, meta_csv_path):
    file_info_validated_path = output_path + "file_info_validated.json"

    # Load info about files already validated (i.e., with path found)
    if Path(file_info_validated_path).is_file():
        file_info_validated = load_json(file_info_validated_path)
    else:
        file_info_validated = {}

    # Add and Validate remaining files
    file_info_validated = build_file_info_validated(output_path, file_info_validated, jsonl_path, meta_csv_path)

    # Save file_info_validated
    save_json(file_info_validated, file_info_validated_path)

    # Read all files in file_info_validated
    documents = read_jsonl_files(
        file_info_validated,
    )

    return documents, file_info_validated


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract molecules and topics from scientific papers."
    )

    parser.add_argument("--input_path", type=str, default="./data/results/")
    parser.add_argument("--output_path", type=str, default="./data/results/")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--n_doc_debug", type=int, default=5)
    # 使用 parse_known_args 避免被父进程（run_pipeline.py）的参数污染 sys.argv
    args, _ = parser.parse_known_args(argv)
    return args


# Main execution
def main(jsonl_path, md_path, meta_csv_path, argv=None):
    args = parse_args(argv)

    # Manage results folder (create a new output folder if it does not exist)
    manage_res_fld(args.output_path)

    # Get documents (retrieve the papers's text) [# Intermediate output: file_info_validated.json]
    documents, file_info = get_documents(args.output_path, jsonl_path, meta_csv_path)

    # Reduce dataset for debugging
    if args.debug:
        documents = {
            k: v for i, (k, v) in enumerate(documents.items()) if i < args.n_doc_debug
        }
        file_info = {
            k: v for i, (k, v) in enumerate(file_info.items()) if i < args.n_doc_debug
        }

    # Assign tags to papers (including molecules) [# Intermediate output: file_info_validated_tags.json, tags/ folder]
    file_info_tags = get_tags(
        documents=documents,
        file_info=file_info,
        output_path=args.output_path,
        md_path=md_path,
        data_path=args.data_path,
    )
    print(f"Successfully tagged {len(file_info_tags)} files")

    # Assign or retrieve canonical SMILES [# Intermediate output: file_info_validated_tags_canon.json]
    file_info_tags_canon = get_canon_smiles(
        file_info_tags,
        output_path=args.output_path,
        data_path=args.data_path,
        debug=args.debug,
        n_doc_debug=args.n_doc_debug,
    )
    print("Assigned tags")

    # Build lookup dictionaries [# Intermediate output: doi2smiles.json, smiles2doi.json]
    build_lookup_dicts(
        file_info_tags_canon,
        output_path=args.output_path,
        debug=args.debug,
        n_doc_debug=args.n_doc_debug,
    )
    print("Build lookup")

    # Prints summary of molecules in database and creates molecules dictionary [# Output: molecules_detailed_info.json]
    molecules = compute_graph_stats(
        save_path=args.output_path,
        save_molecules_dict=True,
    )


if __name__ == "__main__":
    main()
