# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pathlib
from typing import List, Optional

import jsonlines
import pandas as pd
import streamlit as st


repo_path: str = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
DEFAULT_LOG_DIR_1 = os.path.join(repo_path, "logs")
DEFAULT_LOG_DIR_2 = os.path.join(repo_path, "logs")

st.set_page_config(layout="wide")

st.title("Experimental Results")


def get_exp_options(log_dir: str) -> List[str]:
    options: List[str] = [""]
    if os.path.exists(log_dir):
        for exp_dir in os.listdir(log_dir):
            if os.path.isdir(os.path.join(log_dir, exp_dir)):
                options.append(exp_dir)
    options = sorted(options)
    return options


def get_log_dir(base_log_dir: str, exp_name: str) -> str:
    log_dir = os.path.join(base_log_dir, exp_name)
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        return ""
    return log_dir


def get_metrics_table(log_dir: str) -> Optional[pd.DataFrame]:
    filepath = os.path.join(log_dir, "metrics.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        return None


def get_qas_table(log_dir: str, num_id: int, testing_suite_size: int) -> Optional[pd.DataFrame]:
    filepath = os.path.join(log_dir, "QAS.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, sep="|")
        df = df.rename(
            columns={
                "Answer": f"Exp{num_id}",
                "ExactMatch": f"Exp{num_id}-EM",
                "F1": f"Exp{num_id}-F1",
            },
        )
        return df.iloc[:testing_suite_size]
    return None


def join_qas_table(df1: Optional[pd.DataFrame], df2: Optional[pd.DataFrame]) -> List[list]:
    case_list: List[list] = [
        ["Exp1 > Exp2", pd.DataFrame(), None],
        ["Exp1 < Exp2", pd.DataFrame(), None],
        ["Both EM", pd.DataFrame(), None],
        ["Both Mismatch", pd.DataFrame(), None],
        ["Total", pd.DataFrame(), None],
    ]

    if df1 is not None and df2 is not None:
        df2 = df2.drop(columns=["Label"])
        df = pd.merge(df1, df2, left_on="Question", right_on="Question")
        case_list[0][1] = df.loc[df["Exp1-EM"] == 1].loc[df["Exp2-EM"] == 0]
        case_list[1][1] = df.loc[df["Exp1-EM"] == 0].loc[df["Exp2-EM"] == 1]
        case_list[2][1] = df.loc[df["Exp1-EM"] == 1].loc[df["Exp2-EM"] == 1]
        case_list[3][1] = df.loc[df["Exp1-EM"] == 0].loc[df["Exp2-EM"] == 0]
        case_list[4][1] = df

    elif df1 is not None:
        case_list[2][1] = df1.loc[df1["Exp1-EM"] == 1]
        case_list[3][1] = df1.loc[df1["Exp1-EM"] == 0]
        case_list[4][1] = df1

    elif df2 is not None:
        case_list[2][1] = df2.loc[df2["Exp2-EM"] == 1]
        case_list[3][1] = df2.loc[df2["Exp2-EM"] == 0]
        case_list[4][1] = df2

    return case_list


def get_qas_data_to_show(case_list: List[list], has_exp_1: bool, has_exp_2: bool) -> pd.DataFrame:
    show_list = [df_case for _, df_case, to_show in case_list[:-1] if to_show]
    df = pd.concat(show_list)

    if len(df) == 0:
        return df

    column_list = ["Question", "Label"]
    if has_exp_1:
        column_list.append("Exp1")
    if has_exp_2:
        column_list.append("Exp2")

    df = df[column_list].sort_index()
    return df


def get_json_objs(log_dir: str, testing_suite_size: int) -> List[dict]:
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        return None

    filepaths = [os.path.join(log_dir, filename) for filename in os.listdir(log_dir) if filename.endswith("jsonl")]
    if len(filepaths) == 0:
        return None

    with jsonlines.open(filepaths[0], "r") as reader:
        results = [qa for qa in reader]
    return results[:testing_suite_size]


# Selecting experiments
with st.container():
    base_log_dir_1: str = st.text_input(label="Log dir 1", value=DEFAULT_LOG_DIR_1)
    options_1: List[str] = get_exp_options(base_log_dir_1)
    exp_name_1: str = st.selectbox(label="Exp1", options=options_1)
    log_dir_1: str = get_log_dir(base_log_dir_1, exp_name_1)
    testing_suite_size_1: int = st.number_input(label="Testing Suite Size 1(int)", value=500)

    base_log_dir_2: str = st.text_input(label="Log dir 2", value=DEFAULT_LOG_DIR_2)
    options_2: List[str] = get_exp_options(base_log_dir_2)
    exp_name_2: str = st.selectbox(label="Exp2", options=options_2)
    log_dir_2: str = get_log_dir(base_log_dir_2, exp_name_2)
    testing_suite_size_2: int = st.number_input(label="Testing Suite Size 2(int)", value=500)

# Metrics Table
with st.container():
    st.header("Metrics")

    _, metric_1, metric_2, _ = st.columns([1, 4, 4, 1])

    with metric_1:
        st.markdown(f"### Exp1: {exp_name_1}")
        df_metrics_1: Optional[pd.DataFrame] = get_metrics_table(log_dir_1)
        if df_metrics_1 is None:
            st.write("Can't find file metrics.csv")
        st.table(df_metrics_1)

    with metric_2:
        st.markdown(f"### Exp2: {exp_name_2}")
        df_metrics_2: Optional[pd.DataFrame] = get_metrics_table(log_dir_2)
        if df_metrics_2 is None:
            st.write("Can't find file metrics.csv")
        st.table(df_metrics_2)

# Comparison -- QAS Information
with st.container():
    st.header("Question - Answer - Exact Match Score")

    df_qas_1 = get_qas_table(log_dir_1, 1, testing_suite_size_1)
    df_qas_2 = get_qas_table(log_dir_2, 2, testing_suite_size_2)
    if df_qas_1 is None:
        st.write("Can't get Exp1 QAS.csv")
    if df_qas_2 is None:
        st.write("Can't get Exp2 QAS.csv")

    case_list = join_qas_table(df_qas_1, df_qas_2)

    _, qas_case, qas_count, _ = st.columns([1, 1, 1, 1])
    with qas_case:
        st.markdown("### Case")
        for i, (case_label, df_case, _) in enumerate(case_list[:-1]):
            case_list[i][2] = st.checkbox(label=case_label, value=True)
        st.text(case_list[-1][0])

    with qas_count:
        st.markdown("### Count")
        for _, df_case, _ in case_list:
            st.text(len(df_case))

    # QAS Information for chosen records
    df_show = get_qas_data_to_show(case_list, df_qas_1 is not None, df_qas_2 is not None)

    with st.container():
        st.dataframe(df_show, use_container_width=True)

# Comparison -- Json Details
with st.container():
    st.header("Json Details")

    results_1 = get_json_objs(log_dir_1, testing_suite_size_1)
    results_2 = get_json_objs(log_dir_2, testing_suite_size_2)

    num_data: int = len(case_list[-1][1])
    if num_data > 0:
        qa_idx: int = st.number_input(label="Question Index", min_value=0, max_value=num_data - 1, step=1, value=0)
        if 0 <= qa_idx < num_data:
            question: str = case_list[-1][1].iloc[qa_idx]["Question"]
            st.text(f"Question: {question}")

            res_col_1, res_col_2 = st.columns(2)
            res_col_1.markdown(f"### Exp1: {exp_name_1}")
            res_col_2.markdown(f"### Exp2: {exp_name_2}")

            if results_1 is not None:
                res_col_1.json({"Answer": results_1[qa_idx]["answer"]})
                res_col_1.json({"Rationale": results_1[qa_idx]["answer_metadata"]["rationale"]})
                res_col_1.json(results_1[qa_idx]["answer_metadata"]["decomposition_infos"])
            else:
                res_col_1.write("Can't get jsonl file")

            if results_2 is not None:
                res_col_2.json({"Answer": results_2[qa_idx]["answer"]})
                res_col_2.json({"Rationale": results_2[qa_idx]["answer_metadata"]["rationale"]})
                res_col_2.json(results_2[qa_idx]["answer_metadata"]["decomposition_infos"])
            else:
                res_col_2.write("Can't get jsonl file")
