import gzip
import json
import os
import re
import time
import unicodedata
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from ftplib import FTP
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
# from enum import Enum
from typing import Literal, Optional

import pandas as pd
import pubchempy as pcp
import tiktoken
import base64
from molecule.llm_judge.code.llm_lib import parallel_llm_queries
from pydantic import BaseModel, Field, conlist
from rdkit import Chem
from tqdm import tqdm
from openai import OpenAI

# from typing_extensions import Literal
from molecule.llm_judge.code.utils import load_json, sanitize_filename, save_json #, get_client

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MD_FOLDERS = [
    "/llm_data/papers/battery_paper_9k/10k_battery_paper_rename",
    "/llm_data/papers/battery_paper_9k/ASSB_NIBNMB_papers_extracted_md",
    "/llm_data/patents/patent_pdfs_71k_extracted_md_061525_Yao",
    "/llm_data/papers/battery_paper_9k/battery_electrode_process_258_2020_2025_extracted_md",
    "/llm_data/papers/battery_paper_9k/sep_cathode_anode_1417_2015_2025_extracted_md",
    ]

BatteryTopics = Literal[
    "lithium_ion",
    "sodium_ion",
    "lithium_sulfur",
    "potassium_ion",
    "zinc_air",
    "flow_battery",
    "lithium_metal",
    "solid_state",
    "electrolyte",
    "anode",
    "cathode",
    "capacity",
    "cycle_life",
    "energy_density",
    "power_density",
    "safety",
    "thermal_stability",
    "mechanical_stability",
    "fast_charging",
    "cost",
    "sustainability",
    "recycling",
    "manufacturing",
    "battery_management",
    "electric_vehicles",
    "grid_storage",
    "wearable_electronics",
    "life_cycle_assessment",
    "machine_learning",
    "electrochemical_kinetics",
    "interface_stability",
    "nanotechnology",
    "electrode_design",
]


def build_examples(
    n_examples=4,
    data_path="./data/",
):
    path_to_examples = data_path + "smiles.xlsx"
    target_roles = ["solvent", "diluent", "additive"]

    df = pd.read_excel(path_to_examples)

    example = "Here are some examples of valid Molecule() classes with valid 'name', 'role', 'formula', 'smiles' and 'abbreviation' attributes:\n\n"
    dfs = []
    for role in target_roles:
        dfs.append(df[df["type"] == role][:n_examples])
    df = pd.concat(dfs, ignore_index=True)
    for e in range(len(df)):
        name = df["solvent_name"][e]
        role = df["type"][e]
        formula = df["formula\n(PubChem)"][e]
        smiles = df["smile (PubChem)"][e]
        abbreviation = df["abbreviations"][e]
        example += f"Molecule(name='{name}', role='{role}', formula='{formula}', smiles='{smiles}', abbreviation='{abbreviation}')\n"
    return example + "\n"


class Topic(BaseModel):
    name: BatteryTopics = Field(
        ..., description="A major topic discussed in the document"
    )  # BatteryTopics


MoleculeRole = Literal["solvent", "diluent", "additive", "salt", "other"]


class Molecule(BaseModel):
    name: str = Field(
        ...,
        description="The molecule's name.",
    )
    role: MoleculeRole = Field(
        ...,
        description="The molecule's role in the electrolyte. Set to 'other' if not provided.",
    )
    formula: Optional[str] = Field(
        None, description="The molecule's formula. Do not guess if not provided."
    )
    smiles: Optional[str] = Field(
        None, description="The molecule's SMILES string. Do not guess if not provided."
    )
    abbreviation: Optional[str] = Field(
        None, description="The molecule's abbreviation. Do not guess if not provided."
    )
    concentration: Optional[str] = Field(
        None,
        description="The molecule's concentration used (e.g., 1 M, 1.51 mol/L, 0.5 mol/kg, 0.3 m). Do not guess if not provided.",
    )


class ArticleInfo(BaseModel):
    molecules: list[Molecule] = Field(
        ...,
        description=(
            "An exhaustive list of the unique molecules mentioned in the user-provided text, "
            "ordered from the most to the least relevant. Only use curly braces to format your response and nothing else."
        ),
    )
    topics: list[Topic] = Field(
        ...,
        description="A list of the 1-6 topics that best describe what the user-provided text is about, from the most to the least relevant",
    )


def get_max_ctx_win(model):
    windows = {
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4.1-mini-2025-04-14": 1047576,
        "o4-mini-2025-04-16": 200000,
        "gpt-5-mini": 128000
    }
    return windows[model]


def trim_to_token_limit(document, model, system_prompt="", log_fld=None):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(document.text)
    max_tokens = get_max_ctx_win(model) - len(encoding.encode(system_prompt))

    if len(tokens) > max_tokens:
        trimmed_tokens = tokens[:max_tokens]
        document.text = encoding.decode(trimmed_tokens)
        print(f"Retaining only {len(trimmed_tokens)}/{len(tokens)} tokens")
        if log_fld:
            save_json(
                {"n_tokens": len(tokens), "n_trimmed_tokens": len(tokens) - max_tokens},
                log_fld + sanitize_filename(document.doi) + ".json",
            )
    return document.text


### Image processing part
def build_artifacts_map_two_levels(md_folders):
    """
    Scan exactly two levels: md_folder -> subfolder -> *_artifacts
    Returns a dict: { <name_before__artifacts>: <abs_path_to__artifacts> }
    """
    artifacts_map = {}
    if isinstance(md_folders, str):
        md_folders = [md_folders]

    for root in md_folders:
        root = Path(root)
        if not root.is_dir():
            continue

        # First, list subfolders under this md folder
        subfolders = [e for e in os.scandir(root) if e.is_dir()]

        # Progress bar for subfolders in this md folder
        for entry1 in tqdm(subfolders, desc=f"Scanning {root}", unit="folder"):
            try:
                with os.scandir(entry1.path) as it2:
                    for entry2 in it2:
                        if entry2.is_dir() and entry2.name.endswith("_artifacts"):
                            key = entry2.name[:-10]  # strip "_artifacts"
                            artifacts_map[key] = str(Path(entry2.path).resolve())
            except PermissionError:
                continue

    return artifacts_map

### Image processing part
def find_pngs_for_file(file_path, artifacts_map, recursive=False):
    """
    Given 'path/file_name' (any extension allowed), look up 'file_name'
    in artifacts_map and return all .png files inside the corresponding
    _artifacts folder.
    """
    file_path = file_path.replace("-0", "")
    file_name = Path(file_path).stem
    folder = artifacts_map.get(file_name)
    if not folder:
        return []

    pngs = []
    if not recursive:
        try:
            with os.scandir(folder) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".png"):
                        pngs.append(str(Path(e.path).resolve()))
        except FileNotFoundError:
            return []
    else:
        stack = [folder]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        if e.is_dir():
                            stack.append(e.path)
                        elif e.is_file() and e.name.lower().endswith(".png"):
                            pngs.append(str(Path(e.path).resolve()))
            except (PermissionError, FileNotFoundError):
                continue

    pngs.sort()
    return pngs

### Image processing part
def encode_image_to_data_url(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"

def get_tags(
    documents,
    output_path,
    file_info,
    md_path,
    data_path="./data/",
    model="o4-mini-2025-04-16",
    save_results=True,
    save_intermediate_results=True,
    max_workers=None,
    create_logs=True,
    exclude_books=True,
):
    file_path = output_path + "file_info_validated_tags.json"
    if Path(file_path).is_file():
        return load_json(file_path)
    else:
        return assign_tags(
            documents,
            output_path,
            file_info,
            md_path,
            data_path=data_path,
            model=model,
            save_results=save_results,
            save_intermediate_results=save_intermediate_results,
            max_workers=max_workers,
            create_logs=create_logs,
            exclude_books=exclude_books,
        )


def assign_tags(
    documents,
    output_path,
    file_info,
    md_path,
    data_path="./data/",
    model="gpt-4o-mini-2024-07-18",
    save_results=True,
    save_intermediate_results=True,
    max_workers=None,
    create_logs=True,
    exclude_books=True,
):
    def assign_tags_fn(document, image_paths):
        ### Adding images
        image_parts = []
        for image_path in image_paths:
            image_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_image_to_data_url(image_path),
                    "detail": "auto"  # or "low", "high"
                }
            })
        ###
        system_prompt = (
            "You are a battery chemistry expert that specializes in extracting structured information "
            "from unstructured scientific documents provided by the users. "
            "Please extract the requested information from the user-provided text and images. "
            "Only extract pieces of information you are confident about.\n"
        )
        system_prompt = (
            system_prompt + example_str if example_str is not None else system_prompt
        )
        # client = get_client(provider="openai", app="bench")
        client = OpenAI(api_key=OPENAI_API_KEY)

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": trim_to_token_limit(
                            document,
                            model,
                            system_prompt,
                            log_fld,
                        )},
                        # {"type": "text", "text": system_prompt
                        # },
                        *image_parts]  # Unpack the list of image messages
                },
            ],
            response_format=ArticleInfo,
        )

        message = completion.choices[0].message
        if message.parsed:
            n_molecules = len(message.parsed.molecules)
            n_topics = len(message.parsed.topics)
            print(
                f"✅ Successfully assigned {n_molecules} molecules and {n_topics} topics"
            )
            # print(message.parsed.molecules)
            # print(message.parsed.topics)
            tags = {
                "molecules": [mol.__dict__ for mol in message.parsed.molecules],
                "topics": [topic.name for topic in message.parsed.topics],
            }
            if save_intermediate_results:
                save_json(tags, tags_fld + sanitize_filename(document.doi) + ".json")
            return tags
        else:
            raise Exception(
                f"❌ Could not parse the LLM output: {message}. Refusal: {message.refusal}"
            )

    example_str = build_examples(data_path=data_path)
    tags_fld = output_path + "tags/"
    if save_intermediate_results:
        os.makedirs(tags_fld, exist_ok=True)
    if create_logs:
        log_fld = output_path + "logs/"
        os.makedirs(log_fld, exist_ok=True)
    else:
        log_fld = None
    already_tagged = [str(f.stem) for f in Path(tags_fld).glob("*.json")]
    Document = namedtuple("Document", ["doi", "text"])
    DocumentList = [
        Document(doi=doi, text=text)
        for doi, text in documents.items()
        if (sanitize_filename(doi) not in already_tagged)
        and not (doi.startswith("book_") and exclude_books)
    ]
    ### Image processing
    # md_folders = MD_FOLDERS
    md_artifacts_lookup = build_artifacts_map_two_levels(md_path)
    image_paths_each_doi = [find_pngs_for_file(file_info.get(i.doi, None).get("path", None), md_artifacts_lookup) for i in DocumentList]
    ###
    print(
        f"Found {len(DocumentList)} untagged files and {len(already_tagged)} already tagged files."
    )
    print(
        f"Found {sum(bool(sublist) for sublist in image_paths_each_doi)} files have images out of {len(DocumentList)} about-to-tag files"
    )
    results = parallel_llm_queries(
        queries=zip(DocumentList, image_paths_each_doi),
        llm_call_fn=assign_tags_fn,
        llm_args=None,
        llm_kwargs=None,
        max_workers=max_workers if max_workers else int(os.cpu_count() * 0.85),
        max_retries=30,
    )
    if save_intermediate_results:
        print(f"Saved intermediate results in {tags_fld}")

    if save_results:
        all_tagged = [str(f.stem) for f in Path(tags_fld).glob("*.json")]
        for doi in file_info:
            san_doi = sanitize_filename(doi)
            if san_doi in all_tagged:
                d = load_json(tags_fld + san_doi + ".json")
                file_info[doi]["molecules"] = d["molecules"]
                file_info[doi]["topics"] = d["topics"]
            else:
                file_info[doi]["molecules"] = []
                file_info[doi]["topics"] = []
        aggregated_results = output_path + "file_info_validated_tags.json"
        save_json(file_info, aggregated_results)
        print(f"Saved aggregated results in {aggregated_results}")
        return file_info
    return results


def get_remap_dicts(data_path="molecule_extraction/llm_judge_main/data/"):
    def build_abbr_to_can_smiles(fpath):
        abbr_dict = {}
        for _, row in smiles_df.iterrows():
            abbr = row.get("abbreviations")
            if pd.notna(abbr):
                # split on commas, semicolons, or the word "or"
                for token in re.split(r";|\s+or\s+", str(abbr)):
                    token = token.strip()
                    if token:
                        existing = abbr_dict.get(token)
                        if existing is not None:
                            if existing != row["canonical_smile"]:
                                print(
                                    f"Duplicate abbreviation '{token}' with conflicting SMILES: '{existing}' vs '{row['canonical_smile']}'. "
                                    "Keeping first entry.",
                                )
                        else:
                            abbr_dict[token] = row["canonical_smile"]
        save_json(abbr_dict, fpath)
        return abbr_dict

    def get_abbr_to_can_smiles():
        fpath = data_path + "abbr_to_can_smiles.json"
        if Path(fpath).is_file():
            return load_json(fpath)
        else:
            return build_abbr_to_can_smiles(fpath)

    smiles_df = pd.read_excel(data_path + "smiles.xlsx")
    name2cansmiles = dict(zip(smiles_df["solvent_name"], smiles_df["canonical_smile"]))
    pubcsmile2cansmiles = dict(
        zip(smiles_df["smile (PubChem)"], smiles_df["canonical_smile"])
    )
    formula2cansmiles = dict(
        zip(smiles_df["formula\n(PubChem)"], smiles_df["canonical_smile"])
    )
    abb2cansmiles = get_abbr_to_can_smiles()
    remap_dicts = {
        "names": name2cansmiles,
        "pubsmiles": pubcsmile2cansmiles,
        "formulas": formula2cansmiles,
        "abbreviations": abb2cansmiles,
    }
    return remap_dicts  # name2cansmiles, pubcsmile2cansmiles, formula2cansmiles, abb2cansmiles


def replace_greek_letters(molecule: str) -> str:
    greek_letter_mapping = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "ζ": "zeta",
        "η": "eta",
        "θ": "theta",
        "ι": "iota",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "ν": "nu",
        "ξ": "xi",
        "ο": "omicron",
        "π": "pi",
        "ρ": "rho",
        "σ": "sigma",
        "τ": "tau",
        "υ": "upsilon",
        "φ": "phi",
        "χ": "chi",
        "ψ": "psi",
        "ω": "omega",
    }
    for greek, eng in greek_letter_mapping.items():
        molecule = molecule.replace(greek, eng)
    return molecule


def normalize_mol(mol):
    return (
        unicodedata.normalize("NFC", mol)
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
    )


def sanitize_mol_name(mol):
    mol = re.sub(r"\s+", "", mol)
    mol = replace_greek_letters(mol)
    mol = normalize_mol(mol)
    return mol


def find_can_smiles(mol_dict, order, remap_dicts=None):
    for field in order:
        can_smiles = find_can_smiles_mol(
            mol_repr=mol_dict[field],
            repr_type=field,
            remap_dicts=remap_dicts,
        )
        if can_smiles:
            return can_smiles, field
    return None, None


last_call = None


def get_compounds(mol_repr, repr_type, min_delta=0.25):
    csmiles, compounds = None, None
    global last_call
    # print(f"last_call={last_call}")

    # Return the string as it is if it is already a canonical smiles string
    if repr_type == "smiles":
        try:
            csmiles = Chem.CanonSmiles(mol_repr)
            if csmiles == mol_repr:  # The string is already a canonical smiles string
                return csmiles
        except:
            pass

    # If it's not a valid canonical smiles, try to find its smiles string and then canonicalize it

    # Find compound on pubchem
    try:
        if last_call is None:
            last_call = time.time()
        elapsed = time.time() - last_call
        if elapsed < min_delta:
            time.sleep(min_delta - elapsed)
            # print(f"Slept for {min_delta-elapsed} seconds")
        compounds = pcp.get_compounds(mol_repr, repr_type)
        last_call = time.time()
    except:
        last_call = time.time()
        return None

    # If the compound is found and has a valid smiles string, try to canonicalize it
    if compounds and (canon_smiles := compounds[0].canonical_smiles):
        try:
            csmiles = Chem.CanonSmiles(canon_smiles)
            return csmiles
        except:
            return None
    else:
        # If the compound is not found on pubchem, it might still be canonicalized directly with rdkit
        if repr_type == "smiles":
            try:
                csmiles = Chem.CanonSmiles(mol_repr)
                return csmiles
            except:
                return None


def find_can_smiles_mol(
    mol_repr: str,
    repr_type: Literal["name", "smiles", "formula", "abbreviation"] = "name",
    remap_dicts=None,
    data_path="molecule_extraction/llm_judge_main/data/",
):
    if not remap_dicts:
        remap_dicts = get_remap_dicts(data_path=data_path)
    assert repr_type in [
        "name",
        "smiles",
        "formula",
        "abbreviation",
    ], f"Unrecognize representation type {repr_type}"
    match repr_type:
        case "name":
            if not (csmiles := remap_dicts["names"].get(mol_repr, None)):
                csmiles = get_compounds(mol_repr, repr_type)
        case "smiles":
            if not (csmiles := remap_dicts["pubsmiles"].get(mol_repr, None)):
                csmiles = get_compounds(
                    mol_repr, repr_type
                )  # Find compound on pubchem->rdkit
        case "formula":
            if not (csmiles := remap_dicts["formulas"].get(mol_repr, None)):
                csmiles = get_compounds(mol_repr, repr_type)
        case "abbreviation":
            csmiles = remap_dicts["abbreviations"].get(mol_repr, None)
    return csmiles


def get_canon_smiles(
    file_info_tags,
    # order=("name", "smiles", "abbreviation"),
    order=("name", "abbreviation"), ### avoid using LLM extracted smiles per Adam suggestion 9/20.
    output_path="./results/kg/test/",
    data_path="molecule_extraction/llm_judge_main/data/",
    n_threads=int(os.cpu_count() * 0.85),
    parallel=False,  # Cannot be used in parallel due to rate limitations of pubchem API
    debug=False,
    n_doc_debug=5,
):

    def process_molecule(doi, ind_mol, mol_dict):
        can_smiles, found_by = find_can_smiles(mol_dict, order, remap_dicts=remap_dicts)
        return doi, ind_mol, can_smiles, found_by

    output_file = Path(output_path) / "file_info_validated_tags_canon.json"
    if output_file.is_file():
        # print(f"Successfully retrieved {str(output_file)}")
        file_info_tags_canon = load_json(
            output_file
        )  # Contains molecules that have been already canonicalized
    else:
        file_info_tags_canon = {}

    # Identify dois to canonicalize
    doi_already_canononicalized = list(file_info_tags_canon.keys())
    print(f"Already canonicalized dois: {len(doi_already_canononicalized)}")

    doi_to_canonicalize = [
        doi for doi in file_info_tags if doi not in doi_already_canononicalized
    ]
    print(f"Dois to canonicalize: {len(doi_to_canonicalize)}")

    global last_call
    last_call = None

    # Retrieve remapping tables
    remap_dicts = get_remap_dicts(data_path=data_path)

    # Define tasks
    tasks = [
        (doi, ind_mol, mol_dict)
        for ind_file, (doi, info) in enumerate(file_info_tags.items())
        if (info["molecules"] is not None) and (doi in doi_to_canonicalize)
        for ind_mol, mol_dict in enumerate(info["molecules"])
    ]

    # Run parallel loop
    results = []
    if parallel:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(process_molecule, *task) for task in tasks]
            for future in tqdm(
                futures,
                total=len(futures),
                desc="Processing molecules",
            ):
                results.append(future.result())
    else:
        print(f"Processing {len(tasks)} molecules:..")
        for itask, task in enumerate(tasks):
            results.append(process_molecule(*task))
            print(f"{itask} ", end="")
            if (itask + 1) % 20 == 0:
                print()

    # Retrieve results
    mol_counter = Counter()
    n_not_found = 0

    # Retrieve molecules that were already canonicalized
    for doi in doi_already_canononicalized:
        file_info_tags[doi] = file_info_tags_canon[doi]

    # Add molecules that were just canonicalized
    for doi, ind_mol, can_smiles, found_by in results:
        file_info_tags[doi]["molecules"][ind_mol]["can_smiles"] = can_smiles
        file_info_tags[doi]["molecules"][ind_mol]["found_by"] = found_by

    # Save results
    save_json(file_info_tags, output_file)

    # Count how molecules were found
    for doi, doi_dict in file_info_tags.items():
        for mol in doi_dict["molecules"]:
            found_by = mol["found_by"]
            if found_by:
                mol_counter[found_by] += 1
            else:
                n_not_found += 1

    # Print summary of results
    n_found = sum([c for c in mol_counter.values()])
    n_molecules = n_found + n_not_found
    print(f"\nTotal canonical smiles found: {n_found}/{n_molecules}")
    print(f"Molecules were found by:")
    for k, v in mol_counter.items():
        print(f"{k}: {v}")
    return file_info_tags


def build_lookup_dicts(
    file_info_tags,
    output_path,
    debug=False,
    n_doc_debug=5,
):
    doi2smiles = {}
    smiles2dois = {}
    for ind_file, (doi, file_dict) in enumerate(file_info_tags.items()):
        if debug and ind_file >= n_doc_debug:
            break

        # Retrieve molecules with valid canonical smiles
        if mol_dicts := file_dict.get("molecules"):
            found_molecules = list(
                {
                    mol_dict["can_smiles"]
                    for mol_dict in mol_dicts
                    if mol_dict["can_smiles"]
                }
            )
        else:
            found_molecules = []

        # Store retrieved molecules
        if doi in doi2smiles:
            print(f"Found duplicate file doi: {doi}!")
            continue
        doi2smiles[doi] = found_molecules

        # Build smiles2doi dict
        for smiles in found_molecules:
            if smiles not in smiles2dois:
                smiles2dois[smiles] = [doi]
            else:
                smiles2dois[smiles].append(doi)

    save_json(doi2smiles, Path(output_path) / "doi2smiles.json")
    save_json(smiles2dois, Path(output_path) / "smiles2dois.json")
    return


def compute_graph_stats(save_path="./results/kg/test1/",
                        save_molecules_dict=False):
    file_info_tagged = load_json(
        Path(save_path) / "file_info_validated_tags_canon.json")

    # Total number of papers in RAG (books were not tagged)
    n_papers = len(
        [k for k, v in file_info_tagged.items() if not k.startswith("book_")])

    # Total number of molecules that were found by LLM
    n_found_molecules = sum([
        len(doc_dict["molecules"])
        for doc_id, doc_dict in file_info_tagged.items()
    ])

    # Total number of molecules that we could canonicalize
    n_canonicalized = 0
    representation_types = ("name", "smiles", "abbreviation")
    n_found_by = {k: 0 for k in representation_types}
    for doc_id, doc_dict in file_info_tagged.items():
        for mol in doc_dict["molecules"]:
            found_by = mol["found_by"]
            if found_by:
                n_canonicalized += 1
                assert found_by in n_found_by, "Invalid found_by string '{found_by}'"
                n_found_by[found_by] += 1

    # Roles of molecules found
    n_canonicalized = 0
    roles = ("solvent", "diluent", "additive", "salt", "other")
    n_roles = {k: 0 for k in roles}
    n_roles_canonicalized = {k: 0 for k in roles}
    for doc_id, doc_dict in file_info_tagged.items():
        for mol in doc_dict["molecules"]:
            found_by = mol["found_by"]
            role = mol["role"]
            assert role in roles, "Invalid found_by string '{role}'"
            n_roles[role] += 1
            if found_by:
                n_roles_canonicalized[role] += 1
                n_canonicalized += 1

    # Unique molecules
    molecules = {}
    for doc_id, doc_dict in file_info_tagged.items():
        for mol in doc_dict["molecules"]:
            found_by = mol["found_by"]
            csmiles = mol["can_smiles"]
            if found_by:
                if csmiles not in molecules:
                    molecules[csmiles] = {
                        "dois": {doc_id},
                        "mode_role": None,
                        "name": {mol["name"]},
                        "abbreviation": {mol["abbreviation"]},
                        "smiles": {mol["smiles"]},
                        "role_doi": {
                            r: ([doc_id] if r == mol["role"] else [])
                            for r in roles
                        },
                    }
                    molecules[csmiles].update(
                        smiles_to_name(csmiles,
                                       namespace="smiles",
                                       output="dict"))
                else:
                    molecules[csmiles]["dois"].add(doc_id)
                    molecules[csmiles]["name"].add(mol["name"])
                    molecules[csmiles]["abbreviation"].add(mol["abbreviation"])
                    molecules[csmiles]["smiles"].add(mol["smiles"])
                    molecules[csmiles]["role_doi"][mol["role"]].append(doc_id)

    # Postprocess mol dict
    n_roles_unique = {role: 0 for role in roles}
    for mol, mold in molecules.items():
        # Remove null entries
        for rep in representation_types:
            molecules[mol][rep] = [
                r for r in molecules[mol][rep] if r is not None
            ]

        # Convert to list to save the dictionary
        molecules[mol]["dois"] = list(molecules[mol]["dois"])

        # Compute mode counter
        molecules[mol]["roles"] = {
            r: len(molecules[mol]["role_doi"][r])
            for r in roles
        }

        # Compute mode role
        molecules[mol]["mode_role"] = max(molecules[mol]["roles"],
                                          key=molecules[mol]["roles"].get)

        # Count role occurrences
        n_roles_unique[molecules[mol]["mode_role"]] += 1

    print(f"n_papers: {n_papers}")
    if n_papers > 0:
        print(f"n_found_molecules: {n_found_molecules} (AVG number of molecules per paper: {n_found_molecules/n_papers:.3f})")
    else:
        print(f"n_found_molecules: {n_found_molecules} (No papers found)")
    if n_found_molecules > 0:
        print(f"n_canonicalized: {n_canonicalized} (Fraction of canonicalized molecules: {n_canonicalized/n_found_molecules:.3f})")
    else:
        print(f"n_canonicalized: {n_canonicalized} (No molecules found)")
    if n_canonicalized > 0:
        print("canonicalized_by: ")
        for rep in representation_types:
            print(f"{rep}: {n_found_by[rep]} ({n_found_by[rep]/n_canonicalized:.3f}) ")
    if n_found_molecules > 0:
        print("molecule_role: ")
        for role, nr in n_roles.items():
            print(f"{role}: {nr} ({nr/n_found_molecules:.3f}) ")
    if n_canonicalized > 0:
        print("canonicalized_molecule_role: ")
        for role, nr in n_roles_canonicalized.items():
            print(f"{role}: {nr} ({nr/n_canonicalized:.3f}) ")
    n_unique_molecules = len(molecules) if n_canonicalized > 0 else 0
    print(f"n_unique_molecules: {n_unique_molecules}")
    if n_unique_molecules > 0:
        print("unique_molecule_role: ")
        for role, nr in n_roles_unique.items():
            print(f"{role}: {nr} ({nr/n_unique_molecules:.3f}) ")
    else:
        print(f"n_unique_molecules: {n_unique_molecules} (No molecules found)")

    # Save molecules dictionary
    if save_molecules_dict:
        save_json(molecules, Path(save_path) / "molecules_detailed_info.json")
    return molecules


import time
import warnings

import pubchempy as pb
import requests
from rdkit import Chem


def get_title(cid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
    res = requests.get(url)
    data = res.json()
    return data["Record"].get("RecordTitle")


def extract_smiles(compound):
    data = compound.to_dict()
    props = data.get("record", {}).get("props", [])
    smiles_dict = {}

    for prop in props:
        urn = prop.get("urn", {})
        label = urn.get("label", "")
        name = urn.get("name", "")
        value = prop.get("value", {}).get("sval")

        if label == "SMILES" and name == "Absolute":
            smiles_dict["absolute_smiles"] = value
            try:
                canonical_smiles = Chem.CanonSmiles(value)
            except:
                canonical_smiles = None
            smiles_dict["rdkit_canonical_smiles"] = canonical_smiles
        elif label == "SMILES" and name == "Connectivity":
            smiles_dict["connectivity_smiles"] = value
    return smiles_dict


def smiles_to_name(compound_repr, namespace="smiles", output="dict", min_delta=0.25):
    def manage_return(d):
        if output == "dict":
            return d
        else:
            out = []
            for o in output:
                out.append(d[o])
            return out

    assert namespace in ["smiles", "name"], f"Invalid namespace '{namespace}'"
    d = dict(
        common_name=None,
        iupac_name=None,
        synonyms=[],
        rdkit_canonical_smiles=None,
        absolute_smiles=None,
        connectivity_smiles=None,
        cid=None,
    )
    if output != "dict":
        assert all([o in d for o in output]), "Invalid output fields"

    if not isinstance(compound_repr, str):
        return manage_return(d)

    # Find compound
    try:
        time.sleep(
            min_delta
        )  # Ensures that the function is not called more than 4 times per second
        compound = pb.get_compounds(compound_repr, namespace)[0]

        try:
            cid = compound.cid
        except:
            cid = None
        d["cid"] = cid

        if cid:
            # Find title
            try:
                title = get_title(compound.cid)
            except:
                title = None
            d["common_name"] = title

            # Find iupac name
            try:
                iupac_name = compound.iupac_name
            except:
                iupac_name = None
            d["iupac_name"] = iupac_name

            # Find synonyms
            try:
                synonyms = compound.synonyms
            except:
                synonyms = []
            d["synonyms"] = synonyms

            # Find smiles
            if compound:
                d.update(extract_smiles(compound))
        else:
            warnings.warn(
                f"Could not find the compound '{compound_repr}' in the namespace '{namespace}'"
            )

    except:
        warnings.warn(
            f"Could not find the compound '{compound_repr}' in the namespace '{namespace}'"
        )
        compound = None

    return manage_return(d)


from pathlib import Path

import numpy as np
import pandas as pd


def add_names(
    folder="../Ionic_conductivity/raw/",
    debug=False,
    overwrite_existing=False,
    use_unique=True,
):
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.is_file()]
    output_folder = folder.parent / "processed/"
    processed_files = [f for f in output_folder.iterdir() if f.is_file()]
    if not overwrite_existing:
        files = [f for f in files if f.stem not in [pf.stem for pf in processed_files]]
    patterns = ["_smiles", "_sm", "smiles"]
    fields = ["common_name", "iupac_name"]
    for file in files:
        print(f"Processing file '{file}'...")
        df = pd.read_csv(file)
        if debug:
            df = df.head(50)
        col_names = df.columns
        final_col_names = []
        converted_cols = []
        new_df = pd.DataFrame()
        for col_name in col_names:
            final_col_names.append(col_name)
            for pattern in patterns:
                if col_name.endswith(pattern) and col_name not in converted_cols:
                    converted_cols.append(col_name)

                    # Get names
                    if not use_unique:
                        # Old
                        new_cols = df[col_name].apply(
                            lambda x: pd.Series(smiles_to_name(x, output=fields))
                        )
                        new_cols.columns = fields
                    else:
                        # New
                        # Suppose some_function(val) returns [a, b]

                        unique_smiles = df[col_name].unique()

                        # Map each unique value to a tuple/list of two values
                        names_map = {
                            smiles: smiles_to_name(smiles, output=fields)
                            for smiles in unique_smiles
                        }

                        # Create a temporary dataframe
                        new_cols = df[col_name].map(names_map)
                        new_cols = pd.DataFrame(
                            new_cols.tolist(), index=df.index, columns=fields
                        )

                    # Assign to two new columns
                    # df[['col1', 'col2']] = pd.DataFrame(mapped.tolist(), index=df.index)

                    # Build new names
                    if pattern == col_name:
                        join_str = ""
                    else:
                        join_str = "_"
                    for indf, f in enumerate(fields):
                        new_field = col_name.replace(pattern, "") + f"{join_str}{f}"
                        new_df[new_field] = new_cols[f]
                        final_col_names.append(new_field)
        # Append columns
        df[new_df.columns] = new_df

        # Reorder columns
        df = df[final_col_names]

        # Save dataframe
        df.replace({None: np.nan})
        df.to_csv(output_folder / file.name, index=False)
        print("...done!")
