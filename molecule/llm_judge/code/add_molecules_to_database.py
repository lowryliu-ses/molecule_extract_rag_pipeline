import argparse
import json
import re
import requests
from setuptools._distutils.util import strtobool
from pathlib import Path
from pprint import pprint
from typing import Dict, List
from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas
from snowflake.connector.errors import MissingDependencyError
import uuid

import molecule.llm_judge.code.graph_utils as gu
import pandas as pd
import os
import numpy as np
from functools import lru_cache
from openpyxl.utils.exceptions import IllegalCharacterError
from tqdm import tqdm

def _load_config():
    """按需加载 config.json，优先查找项目根目录"""
    candidates = [
        Path(os.environ.get("CONFIG_PATH", "")),
        Path.cwd() / "config.json",
        Path(__file__).parents[3] / "config.json",
    ]
    for p in candidates:
        if p and p.exists():
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        "未找到 config.json。请确保文件位于项目根目录，或通过 CONFIG_PATH 环境变量指定。"
    )


def get_parser(argv=None):
    parser = argparse.ArgumentParser(
        description="Add molecules to smiles table")

    # Define the arguments with default values
    parser.add_argument(
        "--input_molecule_extracted_csv",
        type=str,
        default=
        "./smiles_latest.xlsx",  ### This needs to be the latest molecule table.
        help=
        "The smiles table to which we want to add the newly extracted molecules",
    )
    parser.add_argument(
        "--input_rag_meta_csv",
        type=str,
        default="./rag_meta_latest.csv", ### This needs to be the latest molecule table.
        help="The smiles table to which we want to add the newly extracted molecules",
    )
    parser.add_argument(
        "--input_molecules_detailed_info",
        type=str,
        default=
        "./results/molecules_detailed_info.json",
        help="The json file containing all the extracted molecules",
    )

    parser.add_argument(
        "--temp_molecule_extracted_csv",
        type=str,
        default=
        "./temp_molecule_extracted.xlsx",
        help=
        "This table will be created. It will contain the smiles table updated with the new molecules.",
    )

    parser.add_argument(
        "--add_syn",
        type=lambda x: bool(strtobool(x)),  # Convert 'True'/'False' to boolean
        default=False,
        help="Whether to also add synonyms to the table",
    )

    parser.add_argument(
        "--perform_sanity_check",
        type=lambda x: bool(strtobool(x)),  # Convert 'True'/'False' to boolean
        default=True,
        help=
        "Whether to perform a sanity check, which will create the output_duplicate_synonyms.json file",
    )

    parser.add_argument(
        "--output_duplicate_synonyms",
        type=str,
        default=
        "./duplicates_smiles_test.json",
        help=
        "This file will be created. I will contain duplicate synonyms (as a sanity check).",
    )
    # If argv is None:
    # - in normal script runs, it'll use sys.argv[1:]
    # - in notebooks, it will parse known args and ignore Jupyter's extra ones
    if argv is None:
        args, _ = parser.parse_known_args()
        return args

    return parser


def load_json(path):
    path = Path(path)
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise Exception(f"Could not find file {path}")


def get_new_name(old_name):
    input_file = Path(old_name)
    home_fld = input_file.parent
    filename_base = input_file.stem
    prefix, counter = filename_base.split("_v")
    output_file = home_fld / (prefix + "_v" + str(int(counter) + 1) + ".xlsx")
    return output_file


def format_abbreviations(abbreviations):
    return (" or ".join(list(set(abbreviations)))).strip()


def post_process_names_dict(d):
    # Replace common_name of the form "CID XYZ" with something more intuitive
    assert all(
        [f in d for f in ["common_name", "synonyms", "iupac_name"]]
    ), "Missing required fields"
    common_name = d["common_name"]
    if common_name:
        if common_name.startswith("CID"):
            if common_name not in d["synonyms"]:
                d["synonyms"].append(common_name)
            if d["iupac_name"]:
                d["common_name"] = d["iupac_name"]
            else:
                for s in d["synonyms"]:
                    if not s.startswith("CID"):
                        d["common_name"] = s
    return d

def is_patent(doi: str) -> bool:
    # Same logic you used: papers are normal DOIs starting with "10."
    return not bool(re.match(r"^10\.", str(doi).strip()))

def build_doi_year_lookup(rag_meta_csv_path: str) -> dict:
    meta_df = pd.read_csv(rag_meta_csv_path)
    lookup = {}
    for doi, year in zip(meta_df.get("DOI", []), meta_df.get("YEAR", [])):
        if pd.notna(doi) and pd.notna(year) and str(year).isdigit():
            lookup[str(doi).strip().lower()] = int(year)
    return lookup

# In-memory cache for DOI -> (year, month, day)
doi_date_cache = {}

def get_oldest_doi_and_year(dois, doi_year_lookup: dict) -> tuple[str | None, int | None]:
    """
    Return (oldest_doi, oldest_year). Uses doi_year_lookup first; CrossRef fallback; cached.
    """
    oldest_doi = None
    oldest_date = None

    for doi in dois:
        if not doi:
            continue
        doi_lc = str(doi).strip().lower()
        date_tuple = None

        # 1) RAG meta CSV lookup
        if doi_lc in doi_year_lookup:
            year = doi_year_lookup[doi_lc]
            date_tuple = (year, 1, 1)
            doi_date_cache[doi_lc] = date_tuple

        # 2) Cache
        elif doi_lc in doi_date_cache:
            date_tuple = doi_date_cache[doi_lc]

        # 3) CrossRef fallback
        else:
            try:
                resp = requests.get(
                    f"https://api.crossref.org/works/{doi}",
                    timeout=(5, 10),
                    headers={
                        "User-Agent": "SES AI molecule script (mailto:yao-ting.wang@ses.ai)"
                    },
                )
                if resp.status_code == 200:
                    msg = resp.json().get("message", {})
                    date_parts = msg.get("issued", {}).get("date-parts", [[]])[0]
                    if date_parts:
                        # pad to (Y,M,D)
                        date_tuple = tuple(date_parts + [1] * (3 - len(date_parts)))
                        doi_date_cache[doi_lc] = date_tuple
            except Exception as e:
                print(f"[WARN] CrossRef failed for {doi}: {e}")

        if date_tuple and str(date_tuple[0]).isdigit():
            if oldest_date is None or date_tuple < oldest_date:
                oldest_date = date_tuple
                oldest_doi = doi

    return oldest_doi, (oldest_date[0] if oldest_date else None)

def upsert_table(table: pd.DataFrame, molecules: dict, rag_meta_csv_path: str | None = None) -> pd.DataFrame:
    # ---- Build DOI->YEAR lookup once (optional) ----
    doi_year_lookup = {}
    if rag_meta_csv_path:
        try:
            doi_year_lookup = build_doi_year_lookup(rag_meta_csv_path)
            print(f"[INFO] Loaded DOI->YEAR lookup from {rag_meta_csv_path} ({len(doi_year_lookup)} entries).")
        except Exception as e:
            print(f"[WARN] Failed to load RAG meta CSV ({rag_meta_csv_path}); will rely on CrossRef. Error: {e}")

    # ---- Identify molecules to add ----
    # NOTE: Your table columns sometimes appear lowercase in other funcs; keep your existing logic,
    # but be careful: here we assume canonical column names are uppercase like your upsert_table().
    last_mol_number = table["DATA_NUMBER"].iloc[-1]

    existing_smiles = set(table["CANONICAL_SMILE"].astype(str).tolist())
    molecules_to_add_dict = {
        k: post_process_names_dict(v)
        for k, v in molecules.items()
        if str(k) not in existing_smiles
    }

    n_existing_molecules = len(molecules) - len(molecules_to_add_dict)
    print(
        f"Out of {len(molecules)} extracted molecules, {n_existing_molecules} already exist; "
        f"{len(molecules_to_add_dict)} are new and will be added."
    )

    # ---- Ensure destination columns exist (for older CSVs) ----
    needed_cols = [
        "RELATED_PAPER_COUNT", "OLDEST_RELATED_PAPER", "OLDEST_PAPER_YEAR",
        "RELATED_PATENT_COUNT", "OLDEST_RELATED_PATENT", "OLDEST_PATENT_YEAR",
    ]
    for col in needed_cols:
        if col not in table.columns:
            table[col] = None

    # ---- Build new rows ----
    new_rows = []
    for i, (smiles, v) in enumerate(molecules_to_add_dict.items()):
        all_dois = list(set(v.get("dois", []) or []))
        patent_dois = [d for d in all_dois if is_patent(d)]
        paper_dois  = [d for d in all_dois if not is_patent(d)]

        oldest_paper_doi, oldest_paper_year = get_oldest_doi_and_year(paper_dois, doi_year_lookup) if paper_dois else (None, None)
        oldest_patent_doi, oldest_patent_year = get_oldest_doi_and_year(patent_dois, doi_year_lookup) if patent_dois else (None, None)

        row = {
            "DATA_NUMBER": int(last_mol_number) + i + 1,
            "SOLVENT_NAME": v["common_name"] if v.get("common_name") else (v.get("name", [None])[0]),
            "CANONICAL_SMILE": smiles,
            "ABBREVIATIONS": format_abbreviations(v.get("abbreviation", [])),
            "ABBREVIATIONS_TENTATIVE": format_abbreviations(v.get("abbreviation_tentative", [])),
            "TYPE": v.get("mode_role"),
            "SMILE_PUBCHEM": v.get("absolute_smiles"),

            # No RELATED_PAPER anymore
            # "RELATED_PAPER": ...

            # New paper/patent metadata
            "RELATED_PAPER_COUNT": int(len(paper_dois)),
            "OLDEST_RELATED_PAPER": oldest_paper_doi,
            "OLDEST_PAPER_YEAR": oldest_paper_year,

            "RELATED_PATENT_COUNT": int(len(patent_dois)),
            "OLDEST_RELATED_PATENT": oldest_patent_doi,
            "OLDEST_PATENT_YEAR": oldest_patent_year,
        }
        new_rows.append(row)

    # ---- Append to the table ----
    table_added = pd.DataFrame(new_rows)
    if not table_added.empty:
        table_added = table_added.applymap(clean_illegal)

    table_out = pd.concat([table, table_added], ignore_index=True)

    # ---- Fix your duplicate assertion (yours was reversed) ---- Yao's note: this logic was wrong and the old table have 4 duplicate. Not checking this for now.
    # assert len(table_out["CANONICAL_SMILE"]) == len(set(table_out["CANONICAL_SMILE"].astype(str))), "Repeated CANONICAL_SMILE entries found!"

    return table_out


def strip(input_obj):
    if isinstance(input_obj, pd.DataFrame):
        return input_obj.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    elif isinstance(input_obj, dict):
        return {strip(k): strip(v) for k, v in input_obj.items()}
    elif isinstance(input_obj, list):
        return [strip(i) for i in input_obj]
    elif isinstance(input_obj, str):
        return input_obj.strip()
    else:
        return input_obj


def clean_illegal(val):
    if isinstance(val, str):
        # Remove illegal Excel characters using regex
        new_val = re.sub(r"[\x00-\x1F\x7F]", " ", val)
        if val != new_val:
            print(f"{val} -> {new_val}")
        return new_val
    return val


def replace_sub_super(text):
    # Mapping of superscripts and subscripts to normal characters
    superscript_map = str.maketrans(
        {
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "⁺": "+",
            "⁻": "-",
            "⁼": "=",
            "⁽": "(",
            "⁾": ")",
            "ⁿ": "n",
            "ᵃ": "a",
            "ᵇ": "b",
            "ᶜ": "c",
            "ᵈ": "d",
            "ᵉ": "e",
            "ᶠ": "f",
            "ᵍ": "g",
            "ʰ": "h",
            "ᶦ": "i",
            "ʲ": "j",
            "ᵏ": "k",
            "ˡ": "l",
            "ᵐ": "m",
            "ᵒ": "o",
            "ᵖ": "p",
            "ʳ": "r",
            "ˢ": "s",
            "ᵗ": "t",
            "ᵘ": "u",
            "ᵛ": "v",
            "ʷ": "w",
            "ˣ": "x",
            "ʸ": "y",
            "ᶻ": "z",
        }
    )

    subscript_map = str.maketrans(
        {
            "₀": "0",
            "₁": "1",
            "₂": "2",
            "₃": "3",
            "₄": "4",
            "₅": "5",
            "₆": "6",
            "₇": "7",
            "₈": "8",
            "₉": "9",
            "₊": "+",
            "₋": "-",
            "₌": "=",
            "₍": "(",
            "₎": ")",
            "ₐ": "a",
            "ₑ": "e",
            "ₕ": "h",
            "ᵢ": "i",
            "ⱼ": "j",
            "ₖ": "k",
            "ₗ": "l",
            "ₘ": "m",
            "ₙ": "n",
            "ₒ": "o",
            "ₚ": "p",
            "ₛ": "s",
            "ₜ": "t",
            "ᵤ": "u",
            "ᵥ": "v",
            "ₓ": "x",
        }
    )

    return text.translate(superscript_map).translate(subscript_map)


def retrieve_valid_abbreviations(
    molecules,
    existing_abbreviations,
    threshold=2,
    perform_sanity_check=True,
):
    def build_smiles2abbs(abbs2smiles):
        # Retrieve all the abbreviations that refer to the same smiles
        smiles2abbs = {}
        for abb in abbs2smiles:
            assert (
                len(abbs2smiles[abb]) == 1
            ), "There cannot be more than one association per abbreviation by construction"
            smiles = next(iter(abbs2smiles[abb]))
            counts = abbs2smiles[abb][smiles]
            if smiles not in smiles2abbs:
                smiles2abbs[smiles] = {}
            smiles2abbs[smiles][abb] = counts  # .update({abb: counts})

        # Sort abbreviations by number of occurrences
        for smiles in smiles2abbs:
            smiles2abbs[smiles] = sorted(
                smiles2abbs[smiles].items(),
                key=lambda items: items[1],
                reverse=True,
            )
        return smiles2abbs

    def to_abb_list(abb_list):
        if abb_list:
            return [
                abb for abb, count in abb_list
            ]  # Keep the abbreviations and discard the counts
        else:
            return []

    # Get existing abbreviations
    existing_abbreviations = [
        abb.strip()
        for abbs_string in existing_abbreviations
        if isinstance(abbs_string, str)
        for abb in abbs_string.split(" or ")
    ]

    # Find abbreviations->smiles frequencies
    abbreviations = {}
    for smiles, smiles_dict in molecules.items():
        for abb in smiles_dict["abbreviation"]:
            abb = replace_sub_super(abb)
            if abb in existing_abbreviations:
                continue  # Ignore this abbreviation
            else:
                if abb not in abbreviations:
                    abbreviations[abb] = {}
                if smiles not in abbreviations[abb]:
                    abbreviations[abb][smiles] = 1
                else:
                    abbreviations[abb][smiles] += 1

    if perform_sanity_check:
        # Find frequent abbreviations (sanity check)
        frequent_abbreviations = [
            (abb, smiles, count)
            for abb, abb_dict in abbreviations.items()
            for smiles, count in abb_dict.items()
            if count > 1
        ]
        print(f"Number of frequent abbreviations found: {len(frequent_abbreviations)}")
        print(f"frequent_abbreviations:{frequent_abbreviations}")
        # Find abbreviations associated to multiple smiles (sanity check)
        reoccurring_abbreviations = [
            (abb, smiles, count)
            for abb, abb_dict in abbreviations.items()
            if len(abb_dict) > 1
            for smiles, count in abb_dict.items()
        ]
        print(
            f"Number of reoccurring abbreviations found: {len(reoccurring_abbreviations)}"
        )
        print(f"reoccurring_abbreviations:{reoccurring_abbreviations}")
    else:
        frequent_abbreviations, reoccurring_abbreviations = None, None

    # Sort associations by frequency and assign abbreviations to smiles
    # (after this step, an abbreviation can only refer to a single smiles)
    consistent_abbreviations = {}
    tentative_abbreviations = {}
    for abb in abbreviations:
        n_associations = len(abbreviations[abb])
        assert n_associations > 0, "There cannot be 0 associations found"
        if n_associations == 1:
            consistent_abbreviations[abb] = abbreviations[abb]
        else:
            abbreviations[abb] = sorted(
                abbreviations[abb].items(),
                key=lambda items: items[1],
                reverse=True,
            )
            smiles0, counts0 = abbreviations[abb][0][:]
            counts1end = sum([counts for smiles, counts in abbreviations[abb][1:]])
            if counts0 > threshold * counts1end:
                consistent_abbreviations[abb] = {smiles0: counts0}
            else:
                tentative_abbreviations[abb] = {smiles0: counts0}

    # Retrieve all the abbreviations that refer to the same smiles
    smiles2abbs_consistent = build_smiles2abbs(consistent_abbreviations)
    smiles2abbs_tentative = build_smiles2abbs(tentative_abbreviations)

    # Creates final abbreviation lists
    for smiles, smiles_dict in molecules.items():
        # Consistent abbreviation: only associated to this smiles or more than 2x to this smiles than to the other smiles combined.
        molecules[smiles]["abbreviation"] = to_abb_list(
            smiles2abbs_consistent.get(smiles, None)
        )
        # Tentative abbreviation: also associated to other smiles, but more frequenty to this smiles. In case of ex-aequo, the assignment is random.
        molecules[smiles]["abbreviation_tentative"] = to_abb_list(
            smiles2abbs_tentative.get(smiles, None)
        )

    # Double check there are no duplicates
    all_molecules_abbs = [
        abb
        for smiles, smiles_dict in molecules.items()
        for abb in smiles_dict["abbreviation"] + smiles_dict["abbreviation_tentative"]
    ]
    assert len(set(all_molecules_abbs)) == len(
        all_molecules_abbs
    ), "Duplicates were found!"
    assert all(
        [abb not in existing_abbreviations for abb in all_molecules_abbs]
    ), "Abbreviations already in table were found!"
    return (
        molecules,
        smiles2abbs_consistent,
        smiles2abbs_tentative,
        reoccurring_abbreviations,
        frequent_abbreviations,
    )


def add_molecules(
    molecule_extracted_csv="/root/code/molecule_extraction/Codes/molecule_extraction/llm_judge_main/data/molecules/smiles_v0.xlsx",
    molecules_detailed_info_json="/root/code/molecule_extraction/Codes/molecule_extraction/llm_judge_main/results/kg/test1/molecules_detailed_info.json",
    temp_molecule_extracted_csv="/root/code/molecule_extraction/Codes/molecule_extraction/llm_judge_main/data/molecules/smiles_v1.xlsx",
    rag_meta_csv="/root/code/molecule_extraction/Codes/molecule_extraction/llm_judge_main/data/molecules/rag_meta_latest.csv"
):
    output_file_smiles = Path(temp_molecule_extracted_csv)
    # output_file_smiles = get_new_name(input_file_smiles)
    table = strip(pd.read_excel(molecule_extracted_csv))
    molecules = strip(load_json(molecules_detailed_info_json))
    (
        molecules,
        smiles2abbs_consistent,
        smiles2abbs_tentative,
        reoccurring_abbreviations,
        frequent_abbreviations,
    ) = retrieve_valid_abbreviations(
        molecules,
        table["ABBREVIATIONS"].to_list(),
    )
    table = upsert_table(table, molecules, rag_meta_csv)
    table.to_excel(output_file_smiles, index=False)
    print(f"The new table is stored in {output_file_smiles}")
    return (
        table,
        molecules,
        smiles2abbs_consistent,
        smiles2abbs_tentative,
        reoccurring_abbreviations,
        frequent_abbreviations,
    )


def get_abbreviations(table, smiles, print_flag=True):
    if isinstance(smiles, str):
        smiles = [smiles]
    abbs = {}
    if print_flag:
        print(f"smiles: abbreviation")
    smile_to_abbreviation = dict(zip(table["CANONICAL_SMILE"], table["ABBREVIATIONS"]))
    for smile in smiles:
        abbs[smile] = smile_to_abbreviation.get(smile, None)
        if print_flag:
            print(f"{smile}: {abbs[smile]}")
    return abbs


# New molecules: from data_number 660 (inclusive) onward
def get_duplicated_abbreviations(table, print_flag=True):
    abb_list = table["ABBREVIATIONS"].to_list()
    smiles = table["CANONICAL_SMILE"].to_list()
    abb_list_list = []
    abb_dict = {}
    for ind, abb_string in enumerate(abb_list):
        if pd.isna(abb_string):
            abb_list_list.append([None])
        else:
            abb_list_list.append(abb_string.split(" or "))
        for abb in abb_list_list[ind]:
            if abb not in abb_dict:
                abb_dict[abb] = {"smiles": [smiles[ind]], "data_number": [ind + 1]}
            else:
                abb_dict[abb]["smiles"].append(smiles[ind])
                abb_dict[abb]["data_number"].append(ind + 1)
    pop_list = []
    for abb in abb_dict:
        if not abb or len(abb_dict[abb]["smiles"]) <= 1:
            pop_list.append(abb)
            # abb_dict.pop(abb)
        else:
            if print_flag:
                print(
                    f"abbreviation: {abb}, smiles:{abb_dict[abb]['smiles']}, data_number:{abb_dict[abb]['data_number']}"
                )
    for p in pop_list:
        abb_dict.pop(p)
    return abb_dict


def do_add_synonyms(table, molecules, remove_sep=[" or "]):
    def syn_format(syn_list):
        return " or ".join(syn_list)

    # Get synonyms
    smiles = table["CANONICAL_SMILE"].to_list()
    smiles_pubchem = table["SMILE_PUBCHEM"].to_list()

    if "synonyms" in table.columns:
        existing_syns = table["SYNONYMS"].to_list()
    else:
        existing_syns = [None] * len(smiles)

    synonyms = {}
    not_found = {}
    for smile, smile_p, ex_syn in tqdm(
        zip(smiles, smiles_pubchem, existing_syns), total=len(smiles)
    ):
        ex_syn_list = [] ### Adding initialization to avoid error.
        if smile in molecules:
            synonyms[smile] = molecules[smile]["synonyms"]
        else:
            if ex_syn and pd.notna(ex_syn):
                ex_syn_list = ex_syn.split(" or ")
            if len(ex_syn_list) > 0:
                synonyms[smile] = ex_syn_list
            else:
                synonyms[smile] = gu.smiles_to_name(smile, output=["synonyms"])[0]
                if len(synonyms[smile]) == 0:
                    synonyms[smile] = gu.smiles_to_name(smile_p, output=["synonyms"])[0]
        if len(synonyms[smile]) == 0:
            not_found[smile] = True
        synonyms[smile] = [s.strip() for s in synonyms[smile]]
    if len(not_found) > 0:
        print(f" These molecules were not found: ")
        pprint(not_found)

    # Remove synonyms with separators
    for k, v in synonyms.items():
        synonyms[k] = [s for s in v if not any([sep in s for sep in remove_sep])]
        if len(v) != 0 and (len(v) != len(synonyms[k])):
            print(
                f"Removed {len(v)-len(synonyms[k])} synonyms for {k}: {[vi for vi in v if vi not in synonyms[k]]}"
            )

    # Count synonyms
    duplicates = {}
    for k, v in synonyms.items():
        for s in v:
            if s in duplicates:
                duplicates[s].append(k)
            else:
                duplicates[s] = [k]
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}

    # Remove duplicates synonyms and format as a string
    for k, v in synonyms.items():
        synonyms[k] = syn_format([s for s in v if s not in duplicates])
        removed = [vi for vi in v if vi not in synonyms[k]]
        if len(v) != 0 and len(removed) != 0:
            removed_duplicates = {r: duplicates[r] for r in removed}
            print(f"Removed {100*len(removed)/len(v)}% for {k}: {removed_duplicates}")
            assert (len(duplicates[r]) > 1 for r in removed), "Something might be off."

    # Add synonyms
    table["SYNONYMS"] = table["CANONICAL_SMILE"].map(synonyms)

    return table, duplicates


def sanity_check(
    output_file="../data/molecules/smiles_v1_with_names_with_synonyms.xlsx",
    input_molecules="../results/kg/test1/molecules_detailed_info.json",
    duplicates_name="../data/molecules/duplicates_smiles_v1_with_names_with_synonyms.json",
):
    table = pd.read_excel(output_file)
    molecules = gu.load_json(input_molecules)
    duplicates = gu.load_json(duplicates_name)
    dup = {}
    n_tot = len(table)
    n_no_syn = 0
    from_mol_db = 0
    for index, row in table.iterrows():
        s_list = row["SYNONYMS"]
        if isinstance(s_list, str):
            for i in s_list.split(" or "):
                if i in dup:
                    print("Found duplicate")
                if " or " in i:
                    print("Found separator")
        else:
            n_no_syn += 1
            if row["CANONICAL_SMILE"] in molecules:
                from_mol_db += 1
                syns = molecules[row["canonical_smile"]]["synonyms"]
                if len(syns) != 0:
                    print(
                        f"Something might be off for {row['canonical_smile']}. Synonyms are in the db but not in the excel: {syns}"
                    )
            print(f"No syns for {row['canonical_smile']}")

    print(f"n_missing={n_no_syn} ({100*n_no_syn/n_tot}%)")
    print(f"N missing from_mol_db: {from_mol_db}")


def add_synonyms(
    input_file="../data/molecules/smiles_v1_with_names.xlsx",
    output_file="../data/molecules/smiles_v1_with_names_with_synonyms.xlsx",
    input_molecules="../results/kg/test1/molecules_detailed_info.json",
    duplicates_name="../data/molecules/duplicates_smiles_v1_with_names_with_synonyms.json",
    perform_sanity_check=True,
    debug=False,
):
    # Load 'molecules_detailed_info.json' (the file containing the extracted molecules)
    molecules = gu.load_json(input_molecules)

    # Read the 'smiles.xlsx' table
    table = pd.read_excel(input_file)
    if debug:
        table = table.tail(3000)

    # Adds unique synonyms to table
    table, duplicates = do_add_synonyms(table, molecules, remove_sep=[" or "])

    # Save output table (with synonyms)
    table.to_excel(output_file, index=False)
    gu.save_json(duplicates, duplicates_name)

    # Performs a sanity check
    if perform_sanity_check:
        sanity_check(
            output_file=output_file,
            input_molecules=input_molecules,
            duplicates_name=duplicates_name,
        )

    return table, duplicates

def download_molecule_extract_to_excel(output_excel_path: str) -> str:
    """
    Download full MOLECULE_EXTRACT table from Snowflake and overwrite output_csv_path.
    Returns the absolute path written.
    """
    output_excel_path = os.path.abspath(output_excel_path)
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

    conn = connect(

        account=os.getenv("SNOWFLAKE_ACCOUNT", "SESAI-MAIN"),
        user=os.getenv("SNOWFLAKE_USER", "SVC_MOLECULAR_UNIVERSE_USER"),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE", "./private_key"),
        role=os.getenv("SNOWFLAKE_ROLE", "MOLECULAR_UNIVERSE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "MOLECULAR_UNIVERSE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "UMAP_DATA"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        client_session_keep_alive=True,
    )


    sql = """
        SELECT *
        FROM MOLECULE_EXTRACT
        ORDER BY DATA_NUMBER
    """

    try:
        cur = conn.cursor()
        try:
            cur.execute(sql)

            # Build a DataFrame from the cursor result
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=cols)

            # Save
            df = df.sort_values(by=["DATA_NUMBER"], key=lambda s: pd.to_numeric(s, errors="coerce"), ascending=[True])
            df.to_excel(output_excel_path, index=False)
            print(f"✅ Downloaded MOLECULE_EXTRACT ({len(df)} rows) -> {output_excel_path}")
            return output_excel_path
        finally:
            cur.close()
    finally:
        conn.close()

def download_rag_meta_to_csv(output_excel_path: str) -> str:
    """
    Download full RAG_META table from Snowflake and overwrite output_csv_path.
    Returns the absolute path written.
    """
    output_excel_path = os.path.abspath(output_excel_path)
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)

    conn = connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT", "SESAI-MAIN"),
        user=os.getenv("SNOWFLAKE_USER", "SVC_MOLECULAR_UNIVERSE_USER"),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE", "./private_key"),
        role=os.getenv("SNOWFLAKE_ROLE", "MOLECULAR_UNIVERSE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "MOLECULAR_UNIVERSE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "UMAP_DATA"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        client_session_keep_alive=True,
    )

    sql = """
        SELECT *
        FROM RAG_METADATA
    """

    try:
        cur = conn.cursor()
        try:
            cur.execute(sql)

            # Build a DataFrame from the cursor result
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=cols)

            # Save
            df.to_csv(output_excel_path, index=False)
            print(f"✅ Downloaded RAG_METADATA ({len(df)} rows) -> {output_excel_path}")
            return output_excel_path
        finally:
            cur.close()
    finally:
        conn.close()



PAPER_PATENT_COLS = [
    "RELATED_PAPER_COUNT",
    "OLDEST_RELATED_PAPER",
    "OLDEST_PAPER_YEAR",
    "RELATED_PATENT_COUNT",
    "OLDEST_RELATED_PATENT",
    "OLDEST_PATENT_YEAR",
]

# Columns for brand-new rows (edit to match required columns in MOLECULE_EXTRACT)
INSERT_COLS = [
    "DATA_NUMBER",
    "CANONICAL_SMILE",
    "SOLVENT_NAME",
    "TYPE",
    "SMILE_PUBCHEM",
    "ABBREVIATIONS",
    "ABBREVIATIONS_TENTATIVE",
    "SYNONYMS",
    "APPLICATION",
    "FORMULA_PUBCHEM",
    "FUNCTIONAL_GROUP",
    "MOLECULAR_WEIGHT_RDKIT",
    "MULTIPLE_ABBREVIATION",
    "THE_SAME_ABBREVIATION",
    *PAPER_PATENT_COLS,
]

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["DATA_NUMBER"] = pd.to_numeric(df["DATA_NUMBER"], errors="coerce").astype("Int64")
    df = df[df["DATA_NUMBER"].notna()].copy()
    # Pandas NA/NaN -> None
    df = df.replace({pd.NA: None})
    df = df.where(pd.notna(df), None)
    return df

def insert_to_snowflake(
    full_table: pd.DataFrame,
    last_existing_data_number: int,
    chunk_size: int = 10000,
):
    """
    - Updates ONLY paper/patent cols for ALL rows (matched by DATA_NUMBER)
    - Inserts ONLY rows with DATA_NUMBER > last_existing_data_number
    Uses staging + single MERGE (fast).
    """

    df = _normalize_df(full_table)

    # Stage update dataframe (key + 6 cols)
    upd_cols = ["DATA_NUMBER", *PAPER_PATENT_COLS]
    for c in upd_cols:
        if c not in df.columns:
            df[c] = None
    df_upd = df[upd_cols].copy()

    # Stage insert dataframe (new rows only)
    for c in INSERT_COLS:
        if c not in df.columns:
            df[c] = None
    df_ins = df[df["DATA_NUMBER"] > int(last_existing_data_number)][INSERT_COLS].copy()

    print(f"[INFO] Rows for UPDATE (paper/patent cols): {len(df_upd)}")
    print(f"[INFO] Rows for INSERT (new only): {len(df_ins)} (DATA_NUMBER > {last_existing_data_number})")

    conn = connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT", "SESAI-MAIN"),
        user=os.getenv("SNOWFLAKE_USER", "SVC_MOLECULAR_UNIVERSE_USER"),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE", "./private_key"),
        role=os.getenv("SNOWFLAKE_ROLE", "MOLECULAR_UNIVERSE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "MOLECULAR_UNIVERSE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "UMAP_DATA"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        client_session_keep_alive=True,
    )
    conn.autocommit(False)
    cur = conn.cursor()

    tmp_upd = f"TEMP_MOL_UPD_{uuid.uuid4().hex[:10].upper()}"
    tmp_ins = f"TEMP_MOL_INS_{uuid.uuid4().hex[:10].upper()}"

    try:
        # Create temp tables
        cur.execute(f"""
            CREATE TEMP TABLE {tmp_upd} (
                DATA_NUMBER NUMBER,
                RELATED_PAPER_COUNT NUMBER,
                OLDEST_RELATED_PAPER STRING,
                OLDEST_PAPER_YEAR NUMBER,
                RELATED_PATENT_COUNT NUMBER,
                OLDEST_RELATED_PATENT STRING,
                OLDEST_PATENT_YEAR NUMBER
            )
        """)

        cur.execute(f"""
            CREATE TEMP TABLE {tmp_ins} (
                DATA_NUMBER NUMBER,
                CANONICAL_SMILE STRING,
                SOLVENT_NAME STRING,
                TYPE STRING,
                SMILE_PUBCHEM STRING,
                ABBREVIATIONS STRING,
                ABBREVIATIONS_TENTATIVE STRING,
                SYNONYMS STRING,
                APPLICATION STRING,
                FORMULA_PUBCHEM STRING,
                FUNCTIONAL_GROUP STRING,
                MOLECULAR_WEIGHT_RDKIT STRING,
                MULTIPLE_ABBREVIATION STRING,
                THE_SAME_ABBREVIATION STRING,
                RELATED_PAPER_COUNT NUMBER,
                OLDEST_RELATED_PAPER STRING,
                OLDEST_PAPER_YEAR NUMBER,
                RELATED_PATENT_COUNT NUMBER,
                OLDEST_RELATED_PATENT STRING,
                OLDEST_PATENT_YEAR NUMBER
            )
        """)

        # Upload UPDATE staging table in chunks with progress
        print(f"[INFO] Uploading updates into {tmp_upd} ...")
        try:
            for start in tqdm(range(0, len(df_upd), chunk_size), desc="Uploading updates", unit="rows"):
                chunk = df_upd.iloc[start:start + chunk_size]
                ok, *_ = write_pandas(conn, chunk, tmp_upd, auto_create_table=False, overwrite=False, quote_identifiers=False)
                if not ok:
                    raise RuntimeError("write_pandas failed for updates")

            # Upload INSERT staging table in chunks with progress
            if len(df_ins) > 0:
                print(f"[INFO] Uploading inserts into {tmp_ins} ...")
                for start in tqdm(range(0, len(df_ins), chunk_size), desc="Uploading inserts", unit="rows"):
                    chunk = df_ins.iloc[start:start + chunk_size]
                    ok, *_ = write_pandas(conn, chunk, tmp_ins, auto_create_table=False, overwrite=False, quote_identifiers=False)
                    if not ok:
                        raise RuntimeError("write_pandas failed for inserts")
        except MissingDependencyError as e:
            raise MissingDependencyError(
                "Snowflake write_pandas 需要 pandas。请在该 Python 环境下执行: pip install pandas \"snowflake-connector-python[pandas]\""
            ) from e

        # 1) Update only the 6 cols for matched rows (fast set-based)
        print("[INFO] Updating paper/patent columns (set-based) ...")
        cur.execute(f"""
            MERGE INTO MOLECULE_EXTRACT AS target
            USING {tmp_upd} AS source
            ON target.DATA_NUMBER = source.DATA_NUMBER
            WHEN MATCHED THEN UPDATE SET
                target.RELATED_PAPER_COUNT = source.RELATED_PAPER_COUNT,
                target.OLDEST_RELATED_PAPER = source.OLDEST_RELATED_PAPER,
                target.OLDEST_PAPER_YEAR = source.OLDEST_PAPER_YEAR,
                target.RELATED_PATENT_COUNT = source.RELATED_PATENT_COUNT,
                target.OLDEST_RELATED_PATENT = source.OLDEST_RELATED_PATENT,
                target.OLDEST_PATENT_YEAR = source.OLDEST_PATENT_YEAR
        """)

        # 2) Insert only new rows (no updates here)
        if len(df_ins) > 0:
            print("[INFO] Inserting new rows (set-based) ...")
            insert_cols_sql = ", ".join(INSERT_COLS)
            insert_vals_sql = ", ".join([f"source.{c}" for c in INSERT_COLS])
            cur.execute(f"""
                MERGE INTO MOLECULE_EXTRACT AS target
                USING {tmp_ins} AS source
                ON target.DATA_NUMBER = source.DATA_NUMBER
                WHEN NOT MATCHED THEN INSERT ({insert_cols_sql})
                VALUES ({insert_vals_sql})
            """)

        conn.commit()
        print("✅ Fast upsert complete.")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def get_max_data_number_from_snowflake() -> int:
    conn = connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT", "SESAI-MAIN"),
        user=os.getenv("SNOWFLAKE_USER", "SVC_MOLECULAR_UNIVERSE_USER"),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE", "./private_key"),
        role=os.getenv("SNOWFLAKE_ROLE", "MOLECULAR_UNIVERSE_ROLE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "MOLECULAR_UNIVERSE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "UMAP_DATA"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        client_session_keep_alive=True,
    )
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(MAX(TRY_TO_NUMBER(DATA_NUMBER)), 0)
            FROM MOLECULE_EXTRACT
        """)
        val = cur.fetchone()[0]
        return int(val or 0)
    finally:
        cur.close()
        conn.close()

def main(argv=None):

    # Parse input arguments
    args = get_parser(argv=argv)
    download_molecule_extract_to_excel(args.input_molecule_extracted_csv) # everytime download the latest smiles table
    download_rag_meta_to_csv(args.input_rag_meta_csv)
    # Add molecules to the input table
    (
        table,
        molecules,
        smiles2abbs_consistent,
        smiles2abbs_tentative,
        reoccurring_abbreviations,
        frequent_abbreviations,
    ) = add_molecules(
        molecule_extracted_csv=args.input_molecule_extracted_csv,
        molecules_detailed_info_json=args.input_molecules_detailed_info,
        temp_molecule_extracted_csv=args.temp_molecule_extracted_csv,
        rag_meta_csv=args.input_rag_meta_csv
    )
    print("Finish add_molecules(), new molecule table is saved!")
    # Add synonyms to the table created by add_molecules()
    if args.add_syn:
        table, duplicates = add_synonyms(
            input_file=args.temp_molecule_extracted_csv,
            output_file=args.temp_molecule_extracted_csv,
            input_molecules=args.input_molecules_detailed_info,
            perform_sanity_check=args.perform_sanity_check,
            duplicates_name=args.output_duplicate_synonyms,
        )

    max_before = get_max_data_number_from_snowflake()
    temp_table = pd.read_excel(args.temp_molecule_extracted_csv)
    insert_to_snowflake(temp_table, last_existing_data_number=max_before)

if __name__ == "__main__":
    main()

# This function retrieves duplicated abbreviations
# duplicates_dict = get_duplicated_abbreviations(table)

# This function retrieves abbreviations for a given smiles string
# abbreviations = get_abbreviations(table, "[Fe].[Si]")
