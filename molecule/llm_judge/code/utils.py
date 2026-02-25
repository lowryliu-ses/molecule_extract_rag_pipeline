import datetime
import glob
import json
import os
import re
from pathlib import Path

# from google import genai
# import google.generativeai as genai
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
# from pinecone import Pinecone
from tavily import TavilyClient


def withslash(s):
    return s + "/" if not s.endswith("/") else s


def sanitize_filename(filename: str) -> str:
    # Define invalid characters for Windows and Unix-based systems
    invalid_chars = r'[\/:*?"<>|.]'  # Windows invalid characters
    sanitized_filename = re.sub(
        invalid_chars, "_", filename
    )  # Replace invalid characters with '_'
    return sanitized_filename


def get_timestamp(sanitized=True):
    # Get the current date and time
    current_time = datetime.datetime.now()

    # Format the current date and time into a safe string
    filename = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Sanitize
    if sanitized:
        filename = sanitize_filename(filename)

    return filename


def decompose_path(file_path, get="all"):
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if get == "parent":
        return file_path.parent
    elif get == "stem":
        return file_path.stem
    elif get == "suffix":
        return file_path.suffix
    else:
        return file_path.parent, file_path.stem, file_path.suffix


def parquet_to_excel(folder_path):
    folder = Path(folder_path)

    parquet_files = glob.glob(str(folder / "*.parquet"))

    for parquet_file in parquet_files:
        try:
            # Read the Parquet file into a DataFrame
            df = pd.read_parquet(parquet_file)

            # Convert the Parquet file name to Excel file name
            excel_file = Path(parquet_file).with_suffix(".xlsx")

            # Save the DataFrame to an Excel file
            df.to_excel(excel_file, index=False)

            print(f"Successfully converted {parquet_file} to {excel_file}")
        except Exception as e:
            print(f"Failed to convert {parquet_file}: {e}")


def add_qa_posthoc(
    model_responses_file,
    ground_truth_file="./data/qar_dataset_joah_solvent_diluents_additives_031025_qa_formatted.parquet",
):
    """
    Reads the parquet or excel files containing the model responses (output of answer_sda_questions),
    and adds to the file the the ground-truth answers contained in ground_truth_file

    Example usage:
    files = glob.glob("../results/sda_results_355q_2025-03-31_23-22-16 copy/*")
    for file in files:
        add_qa_posthoc(
            model_responses_file=file,
            ground_truth_file="../data/qar_dataset_joah_solvent_diluents_additives_031025_qa_formatted.parquet",
        )
    """
    gt = pd.read_parquet(ground_truth_file)
    is_parquet = True if model_responses_file.endswith(".parquet") else False

    # Read
    read_fcn = pd.read_parquet if is_parquet else pd.read_excel
    response = read_fcn(model_responses_file)

    # Concatenate
    response = pd.concat([gt, response], axis=1)

    # Write
    if is_parquet:
        response.to_parquet(model_responses_file)
    else:
        response.to_excel(model_responses_file)
    return


def to_path(list_of_paths):
    return [Path(p) for p in list_of_paths]


def get_colors(n_cols=10):
    return plt.cm.tab10.colors


# def get_client(provider="openai", app="mu0"):
#     match provider:
#         case "openai":
#             c = OpenAI
#         case "tavily":
#             c = TavilyClient
#         case "pinecone":
#             c = Pinecone
#         case "google":
#             c = genai.Client
#     match app:
#         case "mu0":
#             get_fun = get_mu0_api
#         case "bench":
#             get_fun = get_api
#     return c(api_key=get_fun(provider=provider))


# def get_mu0_api(provider="openai", path_to_env="~/.config/.mu0_apis.env", api_var=None):
#     if not load_dotenv(dotenv_path=Path(path_to_env).expanduser(), override=True):
#         raise Exception(f"Was not able to find the specified .env file '{path_to_env}'")
#     if not api_var:
#         match provider:
#             case "openai":
#                 api_var = "OPENAI_API_KEY"
#             case "tavily":
#                 api_var = "TAVILY_API_KEY"
#             case "pinecone":
#                 api_var = "PINECONE_API_KEY"
#             case "deepseek":
#                 api_var = "DEEPSEEK_API_KEY"
#             case _:
#                 raise Exception(f"Unrecognized provider '{provider}'")
#     return os.getenv(api_var)


def get_api(provider="deepseek", path_to_env="~/.config/.apis.env", api_var=None):
    if not load_dotenv(dotenv_path=Path(path_to_env).expanduser(), override=True):
        raise Exception(f"Was not able to find the specified .env file '{path_to_env}'")
    if not api_var:
        if provider == "deepseek":
            api_var = "DEEPSEEK_API_KEY"
        elif provider == "openai":
            api_var = "OPENAI_API_KEY"
        elif provider == "pinecone":
            api_var = "PINECONE_API_KEY"
        elif provider == "xai":
            api_var = "XAI_API_KEY"
        elif provider == "lambda":
            api_var = "LAMBDA_API_KEY"
        elif provider == "google":
            api_var = "GEMINI_API_KEY"
        elif provider == "togetherai":
            api_var = "TOGETHER_API_KEY"
        elif provider == "anthropic":
            api_var = "ANTHROPIC_API_KEY"
        else:
            raise Exception(f"Unrecognized API provider '{provider}'")
    return os.getenv(api_var)


def list2str(l, sep=", "):
    return f"{sep}".join(l)


def get_mu0_credentials(type=None, path_to_env="~/.config/.apis.env"):
    if not load_dotenv(dotenv_path=Path(path_to_env).expanduser(), override=True):
        raise Exception(f"Was not able to find the specified .env file '{path_to_env}'")
    usr = os.getenv("MU0_USERNAME")
    pwd = os.getenv("MUO_PWD")
    if type == None:
        return usr, pwd
    if type == "usr":
        return usr
    if type == "pwd":
        return pwd


def read_all_json_files(folder_path, skip=None):
    folder = Path(folder_path)
    json_files = folder.glob("*.json")
    if skip:
        skip_files = {folder / f"{name}.json" for name in skip}
        json_files = [file for file in json_files if file not in skip_files]
    data = {}
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data[file_path.stem] = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {file_path.name}: {e}")

    return data


def save_as_json(filename, variable):
    fld = Path(filename).parent
    os.makedirs(fld, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(variable, f, indent=4)
        print(f"Saved file '{filename}'")
    return


def save_json(d, path):
    with open(path, "w") as f:
        json.dump(d, f, indent=4)


def load_json(path):
    path = Path(path)
    if path.is_file():
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise Exception(f"Could not find file {path}")
