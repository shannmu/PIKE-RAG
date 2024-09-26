# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import importlib
import os
import pickle
import shutil
import yaml

from tqdm import tqdm

from pikerag.document_loaders import get_loader
from pikerag.document_transformers import LLMPoweredRecursiveSplitter
from pikerag.llm_client import BaseLLMClient
from pikerag.utils.config_loader import load_dot_env
from pikerag.utils.logger import Logger
from pikerag.utils.walker import list_files_recursively


def load_yaml_config(config_path: str, args: argparse.Namespace) -> dict:
    with open(config_path, "r") as fin:
        yaml_config: dict = yaml.safe_load(fin)

    # Create logging dir if not exists
    experiment_name = yaml_config["experiment_name"]
    log_dir = os.path.join(yaml_config["log_root_dir"], experiment_name)
    yaml_config["log_dir"] = log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.copy(config_path, log_dir)

    # LLM cache config
    if yaml_config["llm_client"]["cache_config"]["location"] is None:
        yaml_config["llm_client"]["cache_config"]["location"] = os.path.join(log_dir, f"llm_cache.db")
    else:
        yaml_config["llm_client"]["cache_config"]["location"] = os.path.join(
            log_dir,
            yaml_config["llm_client"]["cache_config"]["location"],
        )

    # output doc dir
    output_dir: str = yaml_config["output_doc_directory"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        assert (
            not os.path.isfile(output_dir)
            and len(os.listdir(output_dir)) == 0
        ), f"Output directory {output_dir} not empty!"

    return yaml_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="the path of the yaml config file you want to use")
    # TODO: add more options here, and let the ones in cmd line replace the ones in yaml file
    args = parser.parse_args()

    # Load yaml config.
    yaml_config: dict = load_yaml_config(args.config, args)

    # Load environment variables from dot env file.
    load_dot_env(env_path=yaml_config["dotenv_path"])

    # Create logger instances.
    logger = Logger(name=yaml_config["experiment_name"], dump_folder=yaml_config["log_dir"])
    client_logger = Logger(name="client", dump_mode="a", dump_folder=yaml_config["log_dir"])

    # Dynamically import the chunking protocols
    protocol_module = importlib.import_module(yaml_config["chunking_protocol"]["module_path"])
    chunk_summary_protocol = getattr(protocol_module, yaml_config["chunking_protocol"]["chunk_summary"])
    chunk_summary_refinement_protocol = getattr(protocol_module, yaml_config["chunking_protocol"]["chunk_summary_refinement"])
    chunk_resplit_protocol = getattr(protocol_module, yaml_config["chunking_protocol"]["chunk_resplit"])

    # Initialize the LLM powered Splitter
    llm_client_config = yaml_config["llm_client"]
    client_module = importlib.import_module(llm_client_config["module_path"])
    client_class = getattr(client_module, llm_client_config["class_name"])
    llm_client = client_class(
        logger=client_logger,
        llm_config=yaml_config["llm_client"]["llm_config"],
        **llm_client_config.get("args", {}),
        **yaml_config["llm_client"]["cache_config"],
    )
    assert issubclass(client_class, BaseLLMClient), f"model is not supported for splitting"

    splitter = LLMPoweredRecursiveSplitter(
        llm_client=llm_client,
        first_chunk_summary_protocol=chunk_summary_protocol,
        last_chunk_summary_protocol=chunk_summary_refinement_protocol,
        chunk_resplit_protocol=chunk_resplit_protocol,
        llm_config=yaml_config["llm_client"]["llm_config"],
        logger=logger,
        **yaml_config["splitter"],
    )

    # Documents Loading
    for doc_name, doc_path in tqdm(
        list_files_recursively(directory=yaml_config["input_doc_directory"], extensions=None),
        desc="Loading Files",
    ):
        logger.debug(f"Try loading {doc_name}...")

        # Try get the file loader and load documents
        doc_loader = get_loader(file_path=doc_path, file_type=None)
        if doc_loader is None:
            print(f"Skip file {doc_path} due to undefined Document Loader.")
            continue
        docs = doc_loader.load()

        # Add metadata
        for doc in docs:
            doc.metadata.update(
                {
                    "filename": doc_name,
                }
            )

        # Document Splitting
        documents = splitter.transform_documents(docs)

        # Dump document chunks to disk.
        prefix, _ = os.path.splitext(doc_name)
        output_filepath = os.path.join(yaml_config["output_doc_directory"], f"{prefix}.pkl")
        with open(output_filepath, "wb") as fout:
            pickle.dump(documents, fout)
