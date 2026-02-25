import asyncio
import html
import os
import re

import pandas as pd
import requests
import molecule.llm_judge.code.utils as u

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from openai import OpenAI
# from pinecone import Pinecone, ServerlessSpec
from tavily import TavilyClient

# min number of links to accept 4o-mini-search results
# NUM_LINKS_THRESH = 2
# NUM_TAVILY_SERACH_RESULTS = 3

# index_name = "ses-papers-textbooks-for-rag"
# index = pc.Index(index_name)
# print(index.describe_index_stats())

# data_fld = "./data_mu0/"


def get_acs_citation_from_doi(full_doi, metadata_path, DF_9300, DF_Gyuleen, DF_10K):
    """
    Given a DOI, return an ACS‑style HTML‑formatted citation using the metadata
    from the loaded Excel file. Title is hyperlinked to the DOI.
    Uses 'First Author, et al.' format for multiple authors.
    Always returns a plain Python string.
    """
    if metadata_path.endswith("10K_for_RAG.xlsx"):
        metadata_df = DF_10K
    elif metadata_path.endswith("9300 doi_for_RAG.xlsx"):
        metadata_df = DF_9300
    elif metadata_path.endswith("Gyuleen_update.xlsx"):
        metadata_df = DF_Gyuleen
    else:
        metadata_df = pd.read_excel(metadata_path, engine="openpyxl")
    row = metadata_df[metadata_df["doi"] == full_doi]

    # If DOI not found, return a string (not an HTML object)
    if row.empty:
        return f"DOI {full_doi} not found in metadata."

    row = row.iloc[0]

    # Escape title for safe HTML
    title = (
        html.escape(row["title"]) if pd.notna(row["title"]) else "Title not available"
    )
    doi_url = f"https://doi.org/{full_doi}"
    hyperlinked_title = (
        f'<a href="{doi_url}" target="_blank" rel="noopener noreferrer">' f"{title}</a>"
    )

    # Format authors as a natural list (commas and 'and')
    if pd.notna(row["authors"]):
        author_list = [a.strip() for a in row["authors"].split(";")]
        if len(author_list) == 1:
            authors_formatted = author_list[0]
        elif len(author_list) == 2:
            authors_formatted = f"{author_list[0]} and {author_list[1]}"
        else:
            authors_formatted = ", ".join(author_list[:-1]) + f", and {author_list[-1]}"
    else:
        authors_formatted = "Author unknown"

    journal = row["journal"] if pd.notna(row["journal"]) else "Journal unknown"
    year = str(int(row["year"])) if pd.notna(row["year"]) else "Year unknown"

    # Build ACS‑style citation (string containing HTML markup)
    citation = (
        f"{authors_formatted}. {hyperlinked_title}. "
        f"{journal}, {year}. [DOI: {full_doi}]"
    )
    return citation


def get_doi_dfs(data_fld="./data_mu0/"):
    # Read excel files
    DF_9300 = pd.read_excel(data_fld + "9300 doi_for_RAG.xlsx", engine="openpyxl")
    DF_Gyuleen = pd.read_excel(data_fld + "Gyuleen_update.xlsx", engine="openpyxl")
    DF_10K = pd.read_excel(data_fld + "10K_for_RAG.xlsx", engine="openpyxl")

    # Precompute lowercase DOIs for fast, case-insensitive matching
    DF_9300["doi_lower"] = DF_9300["doi"].str.lower()
    DF_Gyuleen["doi_lower"] = DF_Gyuleen["doi"].str.lower()
    DF_10K["doi_lower"] = DF_10K["doi"].str.lower()
    return DF_9300, DF_Gyuleen, DF_10K


def extract_title_doi_from_filename(filename, data_fld):
    """
    Given a file path from Pinecone, extract the DOI or title,
    fetch the ACS citation if it's a paper, or format a textbook name.
    """
    patterns = [
        r"/llm_data/papers/rag_papers/9300LIB/(.+)-0\.jsonl$",
        r"/llm_data/papers/rag_papers/Gyuleen_update/(.+)-0\.jsonl$",
        r"/llm_data/papers/3p6_jsonl/(.+)-0\.jsonl$",
        r"/llm_data/papers/textbooks_jsonl/(.+)-0\.jsonl$",
    ]

    DF_9300, DF_Gyuleen, DF_10K = get_doi_dfs(data_fld=data_fld)

    for pattern in patterns:
        match = re.search(pattern, filename)
        if not match:
            continue

        extracted = match.group(1)
        full_doi = find_full_doi(
            extracted,
            data_fld + "10K_for_RAG.xlsx",
            DF_9300,
            DF_Gyuleen,
            DF_10K,
        )

        # For the paper DBs, look up the citation
        if "9300LIB" in pattern or "Gyuleen_update" in pattern:
            citation = get_acs_citation_from_doi(
                full_doi,
                data_fld + "10K_for_RAG.xlsx",
                DF_9300,
                DF_Gyuleen,
                DF_10K,
            )
            return f"- {citation}"

        # In the 3p6_jsonl case, the DOI has underscores instead of slashes
        if "3p6_jsonl" in pattern:
            doi = extracted.replace("_", "/")
            citation = get_acs_citation_from_doi(
                full_doi,
                data_fld + "10K_for_RAG.xlsx",
                DF_9300,
                DF_Gyuleen,
                DF_10K,
            )
            return f"- {citation}"

        # For textbooks, just prettify the filename
        if "textbooks_jsonl" in pattern:
            title = extracted.replace("-", " ").title()
            return f"- {title}"

    return None


def find_full_doi(
    doi_suffix: str,
    excel_file_path: str,
    DF_9300,
    DF_Gyuleen,
    DF_10K,
) -> str:
    try:
        # Use preloaded data rather than loading dynamically
        if excel_file_path.endswith("9300 doi_for_RAG.xlsx"):
            df = DF_9300
        elif excel_file_path.endswith("Gyuleen_update.xlsx"):
            df = DF_Gyuleen
        elif excel_file_path.endswith("10K_for_RAG.xlsx"):
            df = DF_10K
        else:
            # Fallback to loading dynamically if an unknown file is requested
            df = pd.read_excel(excel_file_path, engine="openpyxl")
            # Precompute lowercase column if missing
            if "doi_lower" not in df.columns:
                df["doi_lower"] = df["doi"].str.lower()

        # Use precomputed lowercase column for matching
        match = df[df["doi_lower"].str.endswith(doi_suffix.lower(), na=False)]

        if not match.empty:
            return match.iloc[0]["doi"]
        else:
            return manually_get_full_doi(doi_suffix)
    except Exception as e:
        return f"An error occurred: {e}"


def manually_get_full_doi(suffix):
    # DOI prefixes based on common patterns
    prefix_mapping = {
        "D": "10.1039/",
        "j": "10.1016/",
        "/doi.org/10.1007/": "",
        "s": "10.1038/",
        "anie": "10.1002/",
        "aenm": "10.1002/",
        "ange": "10.1002/",
        "chemrxiv": "10.26434/",
        "smsc": "10.1002/",
        "adfm": "10.1002/",
        "1742-6596": "10.1088/",
        "1945-7111": "10.1149/",
        "acsaem": "10.1021/",
        "nsr": "10.1093/",
        "EMD": "10.1109/",
        "elab": "10.1021/",
        "energymater": "10.1016/",
        "acsnano": "10.1021/",
        "s12598": "10.1007/",
        "j.cnki": "10.1007/",
    }

    # Identify the prefix based on the starting pattern
    prefix = next(
        (v for k, v in prefix_mapping.items() if suffix.startswith(k)), "10.1000/"
    )
    # Generate the full DOI
    full_doi = prefix + suffix.replace("/doi.org/", "")
    return full_doi


def get_clients(pc_index_name="ses-papers-textbooks-for-rag"):
    openai_client = u.get_client(provider="openai")
    tavily_client = u.get_client(provider="tavily")
    pc = u.get_client(provider="pinecone")
    index = pc.Index(pc_index_name)
    return openai_client, tavily_client, pc, index


def retrieve_context(
    query,
    top_k_chunks: int = 3,
    rag_enabled: bool = True,
    web_search_enabled: bool = False,
    web_search_client: str = "OpenAI",
    data_fld="./data_mu0/",
    NUM_LINKS_THRESH=2,
    rag_conf=None,
):
    context = ""
    sources = []
    openai_client, tavily_client, pc, index = get_clients(
        pc_index_name=rag_conf["index_name"]
    )

    if web_search_enabled:
        if web_search_client == "Tavily":
            # Offload synchronous Tavily search to the thread pool.
            tavily_response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
                include_raw_content=True,
                include_images=False,
            )
            if tavily_response.get("answer"):
                context += f"Web search result: {tavily_response['answer']}\n\n"
                for result in tavily_response["results"]:
                    title = result["title"]
                    url = result["url"]
                    url_link = f'- Web Search: <a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                    if url_link not in sources:
                        sources.append(url_link)

        elif web_search_client == "OpenAI":
            # Offload OpenAI's synchronous API call to the thread pool.
            completion = openai_client.chat.completions.create(
                model="gpt-4o-mini-search-preview",
                messages=[{"role": "user", "content": query}],
            )
            openai_response = completion.choices[0].message.content
            # Extract links using a regex pattern.
            links = re.findall(r"\(\[([^\]]+)\]\(([^)]+)\)\)", openai_response)
            if len(links) >= NUM_LINKS_THRESH:
                for text, url in links:
                    url_split = url.split("?utm_source=openai")[0]
                    sources.append(f"- {text}: {url_split}")
                context += f"Web search result: {openai_response}\n\n"

    if rag_enabled:
        # Build the payload for your initial retrieval.
        b4_rerank_contexts = []
        if rag_conf["use_search"]:
            # Old search-based
            query_payload = {"inputs": {"text": f"{query}"}, "top_k": 50}
            # Offload the search call to a thread.
            first_retrieval = index.search(
                namespace=rag_conf["namespace"],
                query=query_payload,
            )
            for i, hit in enumerate(first_retrieval["result"]["hits"]):
                b4_rerank_contexts.append(
                    {
                        "id": hit["fields"]["source"],
                        "text": hit["fields"]["context"],
                    }
                )
        else:
            # New query-based
            query_vector = pc.inference.embed(
                model="llama-text-embed-v2",
                inputs=query,
                parameters={"input_type": "passage", "truncate": "END"},
            ).data[0]["values"]
            first_retrieval = index.query(
                namespace=rag_conf["namespace"],
                vector=query_vector,
                top_k=50,
                include_metadata=True,
            )
            for i, hit in enumerate(first_retrieval["matches"]):
                b4_rerank_contexts.append(
                    {
                        "id": hit["metadata"]["source"],
                        "text": hit["metadata"]["context"],
                    }
                )

        # Offload the re-ranking operation to a thread.
        reranked_docs = pc.inference.rerank(
            model="pinecone-rerank-v0",
            query=query,
            documents=b4_rerank_contexts,
            top_n=top_k_chunks,
            return_documents=True,
            parameters={"truncate": "END"},
        )
        # Process each retrieved document.
        for i in range(len(reranked_docs.data)):
            context_text = reranked_docs.data[i]["document"]["text"]
            source_text = reranked_docs.data[i]["document"]["id"]
            # Offload DOI extraction to thread to avoid blocking event loop
            citation = extract_title_doi_from_filename(source_text, data_fld=data_fld)
            if citation:
                context += (
                    f"Database result {i+1}, from {citation[2:]}: {context_text}\n\n"
                )
            else:
                context += f"Database result {i+1}: {context_text}\n\n"
            sources.append(citation)

    # Remove duplicates and None entries from the sources.
    sources = list(dict.fromkeys(s for s in sources if s))
    sources = "\n".join(sources)
    return context, sources
