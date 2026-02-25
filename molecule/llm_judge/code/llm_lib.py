import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import anthropic
# import google.generativeai as genai
import molecule.llm_judge.code.mu0_ask as mu0
import ollama
import openai
import pandas as pd
import molecule.llm_judge.code.prompts_lib as prl
import requests
import molecule.llm_judge.code.utils
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = utils.get_mu0_api(provider="openai")
#openai.api_key = OPENAI_API_KEY

def get_system_message(
    system_msg="You are a knowledgeable and helpful assistant.",
    with_context=False,
    rag_info="./data/sda-rag-info.parquet",
    ind_question=None,
):
    if not with_context:
        return system_msg, {k: "" for k in get_context_fields()}
    else:
        ctx_dict = get_context(ind_question, rag_info)
        system_msg += f"\n\nThe following database results might potentially containt useful information that can help you answer the user query:\n\n{ctx_dict['context']}."
        return system_msg, ctx_dict


def query_ollama_model(
    message,
    model_name="deepseek-r1:1.5b",
    extract_thinking=True,
    strip=True,
    max_n_attempts=5,
    use_chat=False,
    timeout=None,
    rag_on=False,
    ind_question=None,
    rag_info="./data/sda-rag-info.parquet",
):

    model_name = model_name.replace("-rag-on", "")

    # Init variables
    response, thinking = "", ""
    n_attempts = 0

    # Build system message, potentially with rag info
    system_msg, ctx_dict = get_system_message(
        with_context=rag_on, rag_info=rag_info, ind_question=ind_question
    )

    # system_msg = "You are a knowledgeable and helpful assistant."
    # if rag_on:
    #    context, rag_start, rag_end, sources = get_context(ind_question, rag_info)
    #    system_msg += f"\n\nThe following database results might potentially containt useful information that can help you answer the user query:\n\n{context}."

    # Start query loop (keep querying the model for max_n_attempts until you get a valid response)
    while n_attempts < max_n_attempts and response == "":
        llm = ollama.Client(timeout=timeout)
        try:
            if use_chat:  # Use chat model
                response = llm.chat(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_msg,
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ],
                )["message"]["content"]
            else:  # Use base model
                response = llm.generate(
                    model=model_name,
                    prompt=system_msg + "\n" + message,
                )["response"]

            response = "" if response is None else response

            if extract_thinking:
                thinking, response = prl.extract_text_by_tag(
                    response,
                    "think",
                )

            if response == "":
                print(
                    f"Empty string or null model response, attempt: {n_attempts+1}/{max_n_attempts})"
                )

        except Exception as e:
            print(
                f"Ran into an exception while querying the model (response: '{response}', attempt: {n_attempts+1}/{max_n_attempts}), exeption: '{e}'."
            )
        n_attempts += 1

        if strip:
            thinking = thinking.strip()
            response = response.strip()

        # Prepare model output
        if rag_on:
            thinking = prepend_context_to_thinking(thinking, ctx_dict)

    return response, thinking, "", ctx_dict["sources"]


def get_ollama_model_list():
    ollama_client = ollama.Client()
    model_names = [model.model for model in ollama_client.list().models]
    model_names = [re.sub(":latest$", "", name) for name in model_names]
    return model_names


def is_avail_model(model, raise_exception=True):
    model_list = get_ollama_model_list() + get_openai_models()
    model = model.replace("-rag-on", "")
    if model not in model_list:
        model_list_string = ", ".join([f"'{m}'" for m in model_list])
        if raise_exception:
            raise Exception(
                f"Could not find '{model}' among the available Ollama models: [{model_list_string}].\nTry running: 'ollama pull {model}' first."
            )
        else:
            return False
    return True


def get_client(provider="deepseek", api_var=None, api_key=None):
    return OpenAI(
        api_key=(
            api_key if api_key else utils.get_api(provider=provider, api_var=api_var)
        ),
        base_url="https://api.deepseek.com" if provider == "deepseek" else None,
    )


def get_context_fields():
    return ["context", "rag_start", "rag_end", "sources"]


def get_context(ind_question, rag_info):
    fields = get_context_fields()
    # return list(
    #    pd.read_parquet(rag_info)[["context", "rag_start", "rag_end", "sources"]].iloc[
    #        ind_question
    #    ]
    # )
    df = pd.read_parquet(rag_info)
    return {k: df[k].iloc[ind_question] for k in fields}


def prepend_context_to_thinking(thinking, ctx_dict):
    return (
        ctx_dict["rag_start"].strip()
        + "\n"
        + ctx_dict["context"].strip()
        + "\n"
        + ctx_dict["rag_end"].strip()
        + "\n"
        + thinking.strip()
    )


def query_deepseek_model(
    query,
    ind_question=None,
    rag_on=False,
    model_name="deepseek-reasoner",
    rag_info="./data/sda-rag-info.parquet",
):
    model_name = model_name.replace("-rag-on", "")
    model_dict = {
        "deepseek-r1:671b": "deepseek-reasoner",
        "deepseek-r1:671b-r1": "deepseek-reasoner",
        "deepseek-reasoner": "deepseek-reasoner",
        "deepseek-r1:671b-v3": "deepseek-chat",
        "deepseek-chat": "deepseek-chat",
    }
    # Get client
    client = get_client(provider="deepseek")

    # Build system message, potentially with rag info
    system_msg, ctx_dict = get_system_message(
        with_context=rag_on, rag_info=rag_info, ind_question=ind_question
    )

    # Query the model
    response = client.chat.completions.create(
        model=model_dict[model_name],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ],
        stream=False,
    )

    # Extract model answer and thinking
    answer, thinking = (
        response.choices[0].message.content,
        response.choices[0].message.reasoning_content,
    )

    # Prepare model output
    if rag_on:
        thinking = prepend_context_to_thinking(thinking, ctx_dict)

    # Return
    return (
        answer,
        thinking,
        "",
        ctx_dict["sources"],
    )


def query_oai_model(
    query,
    client=None,
    model_name="gpt-4o-2024-08-06",
    system_msg="You are an accurate and helpful assistant.",
    api_var=None,
    api_key=None,
):
    if not client:
        # client = get_client(provider="openai", api_var=api_var, api_key=api_key)
        client = OpenAI(api_key=OPENAI_API_KEY)
    # print(client.api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": query,
            },
        ],
        stream=False,
    )
    # Return response
    return response.choices[0].message.content


def query_openai_model(
    query,
    model_name="gpt-4o-2024-08-06",
    system_msg="You are an accurate and helpful assistant.",
    n_trials=100,
):
    # Check validity of model
    assert (
        model_name in get_openai_models()
    ), f"'model_mame' is an invalid OpenAI model. Allowed models: {', '.join(get_openai_models())}"

    # Query model
    # client = get_client(provider="openai")
    client = OpenAI(api_key=OPENAI_API_KEY)
    trial = 0
    while trial < n_trials:
        try:
            return query_oai_model(
                query=query,
                client=client,
                model_name=model_name,
                system_msg=system_msg,
            )

        except Exception as e:
            trial += 1
            print(f"Retrying ({trial}/{n_trials}) after error:", e)
            time.sleep(2**trial)  # exponential backoff

    return ""


def query_ses_model(
    query,
    model_name="ses-70",
    max_output_length=1024,  # 8192
    local_url="http://localhost:8800/rag",
    rag_on=True,
    search_on=False,
    parse_output=True,
):

    # Headers to indicate the content type is JSON
    headers = {"Content-Type": "application/json"}

    # Data for the POST request
    data = {
        "query": query,
        "maxOutputLength": max_output_length,
        "ragEnabled": rag_on,
        "webSearchEnabled": search_on,
        "webSearchClient": "Tavily",
    }

    # Send the POST request
    response = requests.post(local_url, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code != 200:
        warnings.warn(f"Failed to get response. Status code:", response.status_code)

    if parse_output:
        return parse_ses_response(response)
    else:
        return response


def extract_thinking(text):
    # ? makes is non-greedy (until the first occurrence of the next part of the pattern)
    pattern = r"<\|start_header_id\|>think(.*?)<\|start_header_id\|>answer"
    match = re.search(
        pattern, text, re.DOTALL
    )  # re.DOTALL makes the dot match newline characters
    if match:
        return match.group(1).strip()  # Return the extracted text
    else:
        return None  # Return None if no match is found


def extract_answer(text, extract_sources_f=True):
    # Greedy (until the end of the string)
    pattern = r"<\|start_header_id\|>answer(.*)"
    match = re.search(
        pattern, text, re.DOTALL
    )  # re.DOTALL makes the dot match newline characters
    sources = ""
    if match:
        out = match.group(1).strip()  # Return the extracted text
        if extract_sources_f:
            out, sources = extract_sources(out)
        return out, sources
    else:
        return None, None  # Return None if no match is found


def extract_sources(s):
    sources_pattern1 = "**Sources:**"
    sources_pattern2 = "**Sources**:"

    sources_index = s.find(sources_pattern1)
    if sources_index == -1:
        sources_index = s.find(sources_pattern2)
    if sources_index != -1:
        out_s = s[:sources_index].strip()
        sources = s[sources_index + len(sources_pattern1) :].strip()
    else:
        out_s = s
        sources = ""
    return out_s, sources


def extract_final_answer(text):
    patterns = [None] * 7
    patterns[0] = r"\\boxed\{(.*)\}"
    patterns[1] = r"\*\*Final Answer\*\*:(.*)"
    patterns[2] = r"\*\*Final Answer:\*\*(.*)"
    patterns[3] = r"\*\*Answer\*\*:(.*)"
    patterns[4] = r"\*\*Answer:\*\*(.*)"
    patterns[5] = r"\*\*Conclusion\*\*:(.*)"
    patterns[6] = r"\*\*Conclusion:\*\*(.*)"
    match = None
    for ip, p in enumerate(patterns):
        match = re.search(p, text, re.DOTALL)
        if match:
            outp = match.group(1).strip()
            break
    else:
        outp = None  # Return None if no match is found
    return outp


def parse_ses_response(response, print_f=False):
    if not isinstance(response, str):
        response = response.json()["outputs"]
    thinking = extract_thinking(response)
    answer, sources = extract_answer(response, extract_sources_f=True)
    final_answer = extract_final_answer(answer)
    if print_f:
        print(f"thinking {'-'*40}\n{thinking}\n\n")
        print(f"answer {'-'*40}\n{answer}\n\n")
        print(f"sources {'-'*40}\n{sources}\n\n")
        print(f"final_answer{'-'*40}\n{final_answer}\n\n")

    return answer, thinking, final_answer, sources


def m0_login(
    username=None,
    password=None,
    API_URL="https://prod-api.ses.ai",
    debug=True,
):
    if not username or not password:
        username, password = utils.get_mu0_credentials()

    # Step 1: Authenticate and get the access token
    if debug:
        print(f"→ Logging in as {username}...")
    login_data = {"username": username, "password": password}
    login_headers = {"Content-Type": "application/x-www-form-urlencoded"}

    login_response = requests.post(
        f"{API_URL}/login", data=login_data, headers=login_headers
    )
    if login_response.status_code != 200:
        raise Exception("❌ Failed to log in:", login_response.text)

    token = login_response.json().get("access_token")
    if not token:
        raise Exception("❌ Failed to get access_token")

    if debug:
        print(f"✅ Got token: {token[:8]}…\n")
    return token


def query_mu0_ask_old(
    query,
    model="o4-mini",
    n_rag_chunks=5,
    rag_on=True,
    search_on=False,
    max_output_length=8192,
    username=None,
    password=None,
    API_URL="https://prod-api.ses.ai",
    debug=True,
    return_string=True,
):

    token = m0_login(username=username, password=password, API_URL=API_URL, debug=debug)

    # Step 2: Prepare the /rag request
    rag_payload = {
        "messages": [{"role": "user", "content": query}],
        "maxOutputLength": max_output_length,
        "ragEnabled": rag_on,
        "webSearchEnabled": search_on,
        "webSearchClient": "Tavily",
        "numRagResults": n_rag_chunks,
        "model": model,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if debug:
        print("→ Sending RAG request…")
        print(f"query: {query}\n model: {model}\n n_rag_chunks: {n_rag_chunks}\n")
    rag_response = requests.post(f"{API_URL}/rag", json=rag_payload, headers=headers)

    if rag_response.status_code == 200:
        if debug:
            print("✅ Done.")
        response = rag_response.json()
        if return_string:
            return response["llmOutput"]
        else:
            return response

    else:
        raise Exception("❌ RAG request failed:", rag_response.text)


_token_cache = {
    "token": None,
    "timestamp": 0,
    "ttl": 600,  # 10 minutes in seconds
}
_token_lock = Lock()


def get_timed_token(username, password, API_URL, debug=True):
    now = time.time()
    # First quick check without lock
    if (
        _token_cache["token"] is not None
        and now - _token_cache["timestamp"] <= _token_cache["ttl"]
    ):
        if debug:
            print("✅ Reusing cached token")
        return _token_cache["token"]

    # Use lock to ensure only one thread logs in
    with _token_lock:
        now = time.time()
        if (
            _token_cache["token"] is None
            or now - _token_cache["timestamp"] > _token_cache["ttl"]
        ):
            if debug:
                print("🔐 Logging in (token expired or not found)...")
            token = m0_login(username, password, API_URL, debug)
            _token_cache["token"] = token
            _token_cache["timestamp"] = now

    return _token_cache["token"]


def query_mu0_server(
    query,
    model="o4-mini",
    n_rag_chunks=5,
    rag_on=True,
    search_on=False,
    max_output_length=8192,
    username=None,
    password=None,
    API_URL="https://prod-api.ses.ai",
    debug=True,
    return_string=True,
    rag_conf=None,
):
    def send_request(token):
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        rag_payload = {
            "messages": [{"role": "user", "content": query}],
            "maxOutputLength": max_output_length,
            "ragEnabled": rag_on,
            "webSearchEnabled": search_on,
            "webSearchClient": "Tavily",
            "numRagResults": n_rag_chunks,
            "model": model,
        }
        return requests.post(f"{API_URL}/rag", json=rag_payload, headers=headers)

    token = get_timed_token(username, password, API_URL, debug)

    if debug:
        print("→ Sending RAG request…")
        print(f"query: {query}\n model: {model}\n n_rag_chunks: {n_rag_chunks}\n")

    rag_response = send_request(token=token)

    if rag_response.status_code == 200:
        print("✅ Received valid response.")
        response = rag_response.json()
        if return_string:
            return response["llmOutput"]
        else:
            return response
    elif rag_response.status_code == 400:
        print("❗Query considered not relevant.")
        if return_string:
            return "Irrelevant Query."
        else:
            return rag_response.json()
    else:
        raise Exception("❌ RAG request failed:", rag_response.text)


def query_api(
    query,
    model="o4-mini",
    n_rag_chunks=5,
    rag_on=True,
    search_on=False,
    max_output_length=8192,
    username=None,
    password=None,
    API_URL="https://prod-api.ses.ai",
    debug=True,
    return_string=True,
    rag_conf=None,
):
    if rag_on or search_on:
        # Append rag and/or search results
        query = mu0.build_mu0_prompt(
            query,
            rag_enabled=rag_on,
            numRagResults=n_rag_chunks,
            web_search_enabled=search_on,
            rag_conf=rag_conf,
        )
    return query_model(query, model)


def convert_messages_to_prompt(messages: list[dict]) -> str:
    """
    Converts a list of chat-style messages into a single prompt string.

    Parameters:
        messages (list of dict): Chat messages with 'role' and 'content'.

    Returns:
        str: A flattened prompt string.
    """
    role_prefixes = {"system": "System", "user": "User", "assistant": "Assistant"}

    prompt = ""
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        prefix = role_prefixes.get(role, role.capitalize())
        prompt += f"{prefix}: {content}\n"

    prompt += "Assistant:"  # Let model know it's the assistant's turn
    return prompt


def query_model(
    query,
    model,
    temperature=0,
    system_msg="You are a knowledgeable and helpful assistant.",
    return_thinking=False,
):
    model = alias2model(model)
    provider = model2provider(model)
    base_url = provider2baseurl(provider)
    if model in ["o3-pro", "o3", "o3-mini"]:
        get_api_key = utils.get_mu0_api
    else:
        get_api_key = utils.get_api
    api_key = get_api_key(provider=provider)
    # print(api_key)
    match provider:
        # case "google":
            # client = genai.Client(api_key=api_key)
            # genai.configure(api_key=api_key)
            # client = genai.GenerativeModel(model_name=model)
            # response = client.generate_content(query).text
        case "openai" | "deepseek" | "xai" | "lambda" | "togetherai":
            # client = OpenAI(api_key=api_key, base_url=base_url)
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)
            args = dict(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                ],
                stream=False,
                temperature=temperature,
                model=model,
            )
            if model in ["o3-mini", "o4-mini", "o3", "o3-pro"]:
                args.pop("temperature")
            if model == "o3-pro":
                response = (
                    client.responses.create(
                        model=model,
                        input=convert_messages_to_prompt(args["messages"]),
                        stream=args["stream"],
                    )
                    .output[1]
                    .content[0]
                    .text
                )
            else:
                response = (
                    client.chat.completions.create(**args).choices[0].message.content
                )
            thinking = ""
            if model in [
                "Qwen/Qwen3-235B-A22B-fp8",
                "Qwen/Qwen3-235B-A22B-Thinking-2507",
            ]:
                response, thinking = extract_model_thinking(response)
        case "anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            response = (
                client.messages.create(
                    model=model,
                    system=system_msg,
                    max_tokens=2048,  # required parameter
                    messages=[{"role": "user", "content": query}],
                )
                .content[0]
                .text
            )
        case _:
            raise Exception(f"Unrecognized provider for model '{model}'")
    if isinstance(response, str) and response != "" and response:
        print("✅ Received valid response.")
    if return_thinking:
        return response, thinking
    else:
        return response


def extract_model_thinking(
    response: str,
    start: str = "<think>",
    stop: str = "</think>",
) -> tuple[str, str]:
    # Escape the start and stop tags to avoid regex errors
    start_esc = re.escape(start)
    stop_esc = re.escape(stop)

    # Find the thinking content
    pattern = f"{start_esc}(.*?){stop_esc}"
    think_match = re.search(pattern, response, flags=re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""

    # Remove the thinking content from the response
    final_answer = re.sub(pattern, "", response, flags=re.DOTALL).strip()

    return final_answer, thinking


def test_all_models(
    query="Just say what model you are and who trained you.",
    provider="all",
):

    xai_ms = get_xai_models()
    dseek_ms = get_deepseek_models()
    google_ms = get_google_models()
    lambda_ms = get_lambda_models()
    openai_ms = get_openai_models()
    anthropic_ms = get_anthropic_models()
    togetherai_ms = get_togetherai_models()
    match provider:
        case "google":
            models = google_ms
        case "xai":
            models = xai_ms
        case "deepseek":
            models = dseek_ms
        case "lambda":
            models = lambda_ms
        case "openai":
            models = openai_ms
        case "togetherai":
            models = togetherai_ms
        case "anthropic":
            models = anthropic_ms
        case "all":
            models = (
                google_ms
                + xai_ms
                + dseek_ms
                + lambda_ms
                + openai_ms
                + togetherai_ms
                + anthropic_ms
            )

    for model in models:
        print(f"Querying model '{model}' with query '{query}'...")
        start_time = time.time()
        response = query_model(query=query, model=model)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Model '{model}' took {elapsed:.2f} seconds. Answer: '{response}'\n\n")


def alias2model(model):
    d = {
        "deepseek-r1": "deepseek-reasoner",
        "deepseek-reasoner-base": "deepseek-reasoner",
        "deepseek-reasoner-0528": "deepseek-reasoner",  # From May 28, 2025
    }
    d.update({"4o": "gpt-4o-2024-08-06"})
    d.update(
        {
            "llama-4-maverick": "llama-4-maverick-17b-128e-instruct-fp8",
            "llama-4-scout": "llama-4-scout-17b-16e-instruct",
        }
    )
    d.update({"gemini-2.5-pro": "gemini-2.5-pro-preview-05-06"})
    d.update(
        {
            "qwen3-235B": "Qwen/Qwen3-235B-A22B-fp8",
            "qwen3-235B-thinking": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        }
    )
    d.update({"claude-opus-4": "claude-opus-4-20250514"})
    d.update({"claude-sonnet-4": "claude-sonnet-4-20250514"})
    return d.get(model, model)


def model2provider(model):
    d = {m: "openai" for m in get_openai_models()}
    d.update({m: "deepseek" for m in get_deepseek_models()})
    d.update({m: "google" for m in get_google_models()})
    d.update({m: "lambda" for m in get_lambda_models()})
    d.update({m: "xai" for m in get_xai_models()})
    d.update({m: "togetherai" for m in get_togetherai_models()})
    d.update({m: "anthropic" for m in get_anthropic_models()})
    return d.get(model, None)


def provider2baseurl(provider):
    d = {
        "openai": "https://api.openai.com/v1",
        "xai": "https://api.x.ai/v1",
        "deepseek": "https://api.deepseek.com",
        "lambda": "https://api.lambda.ai/v1",
        "togetherai": "https://api.together.xyz/v1",
    }
    return d.get(provider, None)


def get_mu_ask_models():
    return ["o3-mini", "o4-mini", "o3", "gemini-2.5-pro"]


def get_openai_models():
    return ["o3-pro", "gpt-4o-2024-08-06", "o3-mini", "o4-mini", "o3"]


def get_anthropic_models():
    return ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]


def get_togetherai_models():
    return ["Qwen/Qwen3-235B-A22B-fp8", "Qwen/Qwen3-235B-A22B-Thinking-2507"]


def get_deepseek_models():
    return ["deepseek-reasoner"]


def get_google_models():
    return ["gemini-2.5-pro-preview-05-06", "gemini-2.5-flash"]


def get_lambda_models():
    return ["llama-4-maverick-17b-128e-instruct-fp8", "llama-4-scout-17b-16e-instruct"]


def get_xai_models():
    return ["grok-4-0709", "grok-3-latest", "grok-3-mini-beta"]


def query_mu0_ask(
    query,
    model="o4-mini",
    n_rag_chunks=5,
    rag_on=True,
    search_on=False,
    max_output_length=8192,
    username=None,
    password=None,
    API_URL="https://prod-api.ses.ai",
    debug=True,
    return_string=True,
    use_mu_ask=False,
    rag_conf=None,
):
    if model in get_mu_ask_models() and use_mu_ask:
        query_fun = query_mu0_server
    else:
        query_fun = query_api
    return query_fun(
        query,
        model=model,
        n_rag_chunks=n_rag_chunks,
        rag_on=rag_on,
        search_on=search_on,
        max_output_length=max_output_length,
        username=username,
        password=password,
        API_URL=API_URL,
        debug=debug,
        return_string=return_string,
        rag_conf=rag_conf,
    )


def retry_with_backoff(llm_call_fcn, *args, max_retries=30, **kwargs):
    max_backoff = 2**8  # seconds
    trial = 0
    while trial < max_retries:
        try:
            return llm_call_fcn(*args, **kwargs)
        except Exception as e:
            backoff = min(max_backoff, 2**trial)
            print(
                f"⚠️ Exception occurred (trial {trial}), retrying in {backoff} sec...", e
            )
            time.sleep(backoff)
            trial += 1
    raise RuntimeError(f"❌ Failed after {max_retries} retries.")


def parallel_llm_queries(
    queries,
    llm_call_fn,
    llm_args=None,
    llm_kwargs=None,
    max_workers=10,
    max_retries=30,
):
    """
    Run LLM queries in parallel with retry and backoff.

    Parameters:
        queries (list): list of string queries
        llm_call_fn (function): function to call for each query
        llm_args (tuple): extra positional arguments to pass to the LLM call
        llm_kwargs (dict): extra keyword arguments to pass to the LLM call
        max_workers (int): number of threads
        max_retries (int): retry attempts per query

    Returns:
        list of (query, response) tuples
    """
    max_workers = min(max_workers, os.cpu_count())
    llm_args = llm_args or ()
    llm_kwargs = llm_kwargs or {}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        if isinstance(queries, zip):
            for query, image_paths in queries:
                llm_pos_args = (query, image_paths, *llm_args)
                future = executor.submit(
                    retry_with_backoff,
                    llm_call_fn,
                    *llm_pos_args,
                    max_retries=max_retries,
                    **llm_kwargs,
                )
                futures.append((query, future))
        else:
            for query in queries:
                llm_pos_args = (query, *llm_args)
                future = executor.submit(
                    retry_with_backoff,
                    llm_call_fn,
                    *llm_pos_args,
                    max_retries=max_retries,
                    **llm_kwargs,
                )
                futures.append((query, future))

        for query, future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Failed query: '{query}'", e)
                results.append(f"Error '{e}' with query '{query}'")

    return results
