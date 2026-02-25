from molecule.llm_judge.code.rag_mu0 import retrieve_context


def build_prompt(query, context, sources, rag_enabled, web_search_enabled):
    # Construct the prompt by combining the query and retrieval results
    initial_prompt = f"{query}"
    extended_prompt = ""
    if rag_enabled or web_search_enabled:
        extended_prompt += "\nThe following are"
        if rag_enabled:
            extended_prompt += " database"
        if web_search_enabled:
            if rag_enabled:
                extended_prompt += " and"
            extended_prompt += " internet"
        extended_prompt += (
            " search results you might find useful when answering the query:\n"
        )
        extended_prompt += context
        extended_prompt += "These results may or may not be relevant. "
        extended_prompt += "Think about which parts of this information are useful to answer the query. "
        extended_prompt += "If you name specific molecules in your response, make sure to state the full molecule name before using any abbreviations.\n"
    prompt = initial_prompt + extended_prompt
    return prompt


def query_model(
    prompt,
    model,
    max_outputLength=8192,
):
    response = ""
    return response


# Run retrieval based on the enabled options


def build_mu0_prompt(
    query,
    rag_enabled=True,
    numRagResults=3,
    web_search_enabled=False,
    web_search_client="OpenAI",
    data_fld="./data_mu0/",
    rag_conf=None,
):
    context, sources = retrieve_context(
        query=query,
        top_k_chunks=numRagResults,
        rag_enabled=rag_enabled,
        web_search_enabled=web_search_enabled,
        web_search_client=web_search_client,
        data_fld=data_fld,
        rag_conf=rag_conf,
    )
    prompt = build_prompt(
        query=query,
        context=context,
        sources=sources,
        rag_enabled=rag_enabled,
        web_search_enabled=web_search_enabled,
    )
    return prompt
