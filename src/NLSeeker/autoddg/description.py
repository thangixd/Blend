from __future__ import annotations


# --- UFD --

UFD_SYSTEM_MESSAGE = (
    "You are an assistant for a dataset search engine. Your goal is to improve "
    "the readability of dataset descriptions for dataset search engine users."
)

_UFD_INTRO = (
    "Answer the question using the following information.\n"
    "\n"
    "First, consider the dataset sample:\n"
    "\n"
    "{dataset_sample}\n"
)

_UFD_PROFILE = (
    "Additionally, the dataset profile is as follows:\n"
    "\n"
    "{dataset_profile}\n"
    "\n"
    "Based on this profile, please add sentence(s) to enrich the dataset description.\n"
)

_UFD_SEMANTIC = (
    "Furthermore, the semantic profile of the dataset columns is as follows:\n"
    "{semantic_profile}\n"
    "\n"
    "Based on this information, please add sentence(s) discussing the semantic profile "
    "in the description.\n"
)

_UFD_TOPIC = (
    "Moreover, the dataset topic is: {data_topic}. Based on this topic, please add "
    "sentence(s) describing what this dataset can be used for.\n"
)

_UFD_CLOSING = (
    "Question: Based on the information above and the requirements, provide a dataset "
    "description in sentences. Use only natural, readable sentences without special "
    "formatting.\n"
    "\n"
    "Answer:\n"
)


def build_ufd_prompt(
    *,
    dataset_sample: str,
    dataset_profile: str | None,
    semantic_profile: str | None,
    data_topic: str | None,
    use_profile: bool,
    use_semantic_profile: bool,
    use_topic: bool,
    description_words: int = 100,
) -> str:
    parts: list[str] = [_UFD_INTRO.format(dataset_sample=dataset_sample)]
    if use_profile and dataset_profile:
        parts.append(_UFD_PROFILE.format(dataset_profile=dataset_profile))
    if use_semantic_profile and semantic_profile:
        parts.append(_UFD_SEMANTIC.format(semantic_profile=semantic_profile))
    if use_topic and data_topic:
        parts.append(_UFD_TOPIC.format(data_topic=data_topic))
    parts.append(_UFD_CLOSING)
    parts.append(f"Target length: approximately {int(description_words)} words.")
    return "\n".join(parts)


# --- SFD ---

SFD_SYSTEM_MESSAGE = (
    "You are an assistant for a dataset search engine. Your goal is to improve "
    "the performance of the dataset search engine for keyword queries."
)

SFD_TEMPLATE = (
    "Dataset Overview:\n"
    "- Please keep the exact initial description of the dataset as shown in beginning the prompt.\n"
    "\n"
    "Key Themes or Topics:\n"
    "- Central focus on a broad area of interest (e.g., urban planning, socio-economic factors, environmental analysis).\n"
    "- Data spans multiple subtopics or related areas that contribute to a holistic understanding of the primary theme.\n"
    "Example:\n"
    "- theme1/topic1\n"
    "- theme2/topic2\n"
    "- theme3/topic3\n"
    "\n"
    "Applications and Use Cases:\n"
    "- Facilitates analysis for professionals, policymakers, researchers, or stakeholders.\n"
    "- Useful for specific applications, such as planning, engineering, policy formulation, or statistical modeling.\n"
    "- Enables insights into patterns, trends, and relationships relevant to the domain.\n"
    "Example:\n"
    "- application1/usecase1\n"
    "- application2/usecase2\n"
    "- application3/usecase3\n"
    "\n"
    "Concepts and Synonyms:\n"
    "- Includes related concepts, terms, and variations to ensure comprehensive coverage of the topic.\n"
    "- Synonyms and alternative phrases improve searchability and retrieval effectiveness.\n"
    "Example:\n"
    "- concept1/synonym1\n"
    "- concept2/synonym2\n"
    "- concept3/synonym3\n"
    "\n"
    "Keywords and Themes:\n"
    "- Lists relevant keywords and themes for indexing, categorization, and enhancing discoverability.\n"
    "- Keywords reflect the dataset's content, scope, and relevance to the domain.\n"
    "Example:\n"
    "- keyword1\n"
    "- keyword2\n"
    "- keyword3\n"
    "\n"
    "Additional Context:\n"
    "- Highlights the dataset's relevance to specific challenges or questions in the domain.\n"
    "- May emphasize its value for interdisciplinary applications or integration with related datasets.\n"
    "Example:\n"
    "- context1\n"
    "- context2\n"
    "- context3"
)

_SFD_USER_PROMPT = (
    "You are given a dataset about the topic {topic}, with the following initial description:\n"
    "\n"
    "{initial_description}\n"
    "\n"
    "Please expand the description by including the exact topic. Additionally, add as many "
    "related concepts, synonyms, and relevant terms as possible based on the initial "
    "description and the topic.\n"
    "\n"
    "Unlike the initial description, which is focused on presentation and readability, the "
    "expanded description is intended to be indexed at the backend of a dataset search "
    "engine to improve searchability.\n"
    "\n"
    "Therefore, focus less on readability and more on including all relevant terms related "
    "to the topic. Make sure to include any variations of the key terms and concepts that "
    "could help improve retrieval in search results.\n"
    "\n"
    "Please follow the structure of the following example template:\n"
    "{template}\n"
)


def build_sfd_prompt(*, topic: str, initial_description: str) -> str:
    return _SFD_USER_PROMPT.format(
        topic=topic,
        initial_description=initial_description,
        template=SFD_TEMPLATE,
    )


__all__ = [
    "UFD_SYSTEM_MESSAGE",
    "SFD_SYSTEM_MESSAGE",
    "SFD_TEMPLATE",
    "build_ufd_prompt",
    "build_sfd_prompt",
]
