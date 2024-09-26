# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os
from typing import Dict, List, Tuple, Union

import jsonlines
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt


st.set_page_config(layout="wide")

st.title("Chunks with Atom Questions")


def get_chunks(filepath: str) -> Dict[str, List[Dict[str, Union[str, str, List[str]]]]]:
    chunks_by_title: Dict[str, List[Dict[str, Union[str, str, List[str]]]]] = {}
    if not os.path.exists(filepath):
        return chunks_by_title

    with jsonlines.open(filepath, "r") as reader:
        for chunk_dict in reader:
            title = chunk_dict["title"]
            if title not in chunks_by_title:
                chunks_by_title[title] = []

            chunks_by_title[title].append(
                {
                    "chunk_id": chunk_dict["chunk_id"],
                    "content": chunk_dict["content"],
                    "atom_questions": chunk_dict["atom_questions"],
                }
            )

    return chunks_by_title


def count_chunk_info(chunks_by_title: Dict[str, List[Dict[str, Union[str, str, List[str]]]]]):
    num_title = len(chunks_by_title)
    num_chunk = sum([len(v) for v in chunks_by_title.values()])
    num_question = sum([len(v["atom_questions"]) for vl in chunks_by_title.values() for v in vl])
    content_length = [len(v["content"]) for vl in chunks_by_title.values() for v in vl]
    words_count = [len(v["content"].split(" ")) for vl in chunks_by_title.values() for v in vl]
    return (
        num_title, num_chunk, num_question,
        num_question / num_chunk if num_chunk else None,
        min(content_length) if content_length else None,
        max(content_length) if content_length else None,
        np.mean(content_length) if content_length else None,
        min(words_count) if words_count else None,
        max(words_count) if words_count else None,
        np.mean(words_count) if words_count else None
    )


def get_title_to_show_step1(all_titles: List[str], num_per_page: int=10) -> Tuple[int, List[str]]:
    num_chunks = len(all_titles)
    num_pages = math.ceil(num_chunks / num_per_page)
    st.text(f"{num_chunks} chunks in total, showed in {num_pages} pages")

    title_to_show = []
    if num_pages == 1:
        title_to_show = all_titles
    elif num_pages > 1:
        page_idx = st.select_slider(label="Page Index", options=[1 + i for i in range(num_pages)], value=1)
        title_to_show = all_titles[(page_idx - 1) * 10 : page_idx * 10]

    return num_chunks, title_to_show


def show_details_of_title(title_to_show: List[str], v1_dict: dict, v2_dict: dict):
    for title in title_to_show:
        with st.expander(title):
            col1, col2 = st.columns([1, 1])
            if title in v1_dict:
                col1.json(v1_dict[title])
            if title in v2_dict:
                col2.json(v2_dict[title])


# Selecting experiments
with st.container():
    tag_path_1: str = st.text_input(label="tagging data jsonline file path 1")
    tag_path_2: str = st.text_input(label="tagging data jsonline file path 2")
    chunks_by_title_1 = get_chunks(tag_path_1)
    chunks_by_title_2 = get_chunks(tag_path_2)

    _, stat_name, stat_1, stat_2, _ = st.columns([1, 1, 1, 1, 1])
    with stat_name:
        st.markdown("### Statistics")
        for name in [
            "#Title", "#Chunk", "#Question", "#Question/#Chunk",
            "Min Content", "Max Content", "Mean Content",
            "Min #Word", "Max #Word", "Mean #Word",
        ]:
            st.text(name)
    with stat_1:
        st.markdown("### Version 1")
        res = count_chunk_info(chunks_by_title_1)
        for r in res:
            st.text(r)
    with stat_2:
        st.markdown("### Version 2")
        res = count_chunk_info(chunks_by_title_2)
        for r in res:
            st.text(r)


# Only in Version 1
with st.container():
    st.header("Chunks with Title only in Version 1")
    v1_only = {
        title: values
        for title, values in chunks_by_title_1.items() if title not in chunks_by_title_2
    }
    num_titles, title_to_show = get_title_to_show_step1(all_titles=[title for title in v1_only.keys()])
    show_details_of_title(title_to_show, v1_only, {})


# Only in Version 2
with st.container():
    st.header("Chunks with Title only in Version 2")
    v2_only = {
        title: values
        for title, values in chunks_by_title_2.items() if title not in chunks_by_title_1
    }
    num_titles, title_to_show = get_title_to_show_step1(all_titles=[title for title in v2_only.keys()])
    show_details_of_title(title_to_show, {}, v2_only)


# Both in
with st.container():
    st.header("Chunks in Both Versions")
    num_titles, title_to_show = get_title_to_show_step1(
        all_titles=[title for title in chunks_by_title_1.keys() if title in chunks_by_title_2],
    )
    show_details_of_title(title_to_show, chunks_by_title_1, chunks_by_title_2)


# Chunk length comparison
with st.container():
    st.header("Chunk Length Distribution")
    chunk_length_1 = [len(v["content"]) for vl in chunks_by_title_1.values() for v in vl]
    chunk_length_2 = [len(v["content"]) for vl in chunks_by_title_2.values() for v in vl]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    axes[0].hist(chunk_length_1)
    axes[1].hist(chunk_length_2)
    st.pyplot(fig)
