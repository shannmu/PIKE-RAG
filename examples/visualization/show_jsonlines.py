# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import os

import jsonlines
import streamlit as st


DEFAULT_JSONLINE_PATH = ""

st.set_page_config(layout="wide")

st.title("Experimental Jsonlines")

# Selecting experiments
with st.container():
    jsonl_filepath: str = st.text_input(label="Jsonlines path", value=DEFAULT_JSONLINE_PATH)
    testing_suite_size: int = st.number_input(label="Testing Suite Size (int)", value=500)

    data = []
    if os.path.exists(jsonl_filepath):
        with jsonlines.open(jsonl_filepath, "r") as reader:
            data = [d for d in reader]
    data = data[:testing_suite_size]

# Json Details
with st.container():
    st.header("Json Details")

    expanded = st.checkbox(f"Expanded?")

    num_data = len(data)
    page_size = 10
    num_pages = math.ceil(num_data / page_size)

    if num_pages >= 2:
        page_idx = st.select_slider(label="Page Index", options=[1 + i for i in range(num_pages)], value=1)
    else:
        page_idx = 1

    if num_pages >= 1:
        for idx in range((page_idx - 1) * page_size, min(page_idx * page_size, num_data)):
            item_title = f"Index {idx}"
            if "question" in data[idx]:
                item_title = f"Q: {data[idx]['question']}"
            if "answer_metric_scores" in data[idx] and "ExactMatch" in data[idx]["answer_metric_scores"]:
                if data[idx]["answer_metric_scores"]["ExactMatch"]:
                    item_title = f"[√] {item_title}"
                else:
                    item_title = f"[×] {item_title}"

            with st.expander(item_title):
                st.json(data[idx], expanded=expanded)
