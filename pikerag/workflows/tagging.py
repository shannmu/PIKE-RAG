# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib
import os
from collections import Counter
from typing import List

from langchain_core.documents import Document

from pikerag.document_transformers import LLMPoweredTagger
from pikerag.llm_client import BaseLLMClient
from pikerag.utils.config_loader import load_protocol
from pikerag.utils.logger import Logger


class TaggingWorkflow:
    def __init__(self, yaml_config: dict) -> None:
        self._yaml_config: dict = yaml_config

        self._init_logger()

        self._load_original_documents()

        self._init_tagger()

    def _init_logger(self) -> None:
        self._logger: Logger = Logger(
            name=self._yaml_config["experiment_name"],
            dump_folder=self._yaml_config["log_dir"],
        )

    def _load_original_documents(self) -> None:
        # Dynamically load the document loading function, then load documents
        doc_loading_module = importlib.import_module(self._yaml_config["ori_doc_loading"]["module"])
        doc_loading_func = getattr(doc_loading_module, self._yaml_config["ori_doc_loading"]["name"])
        self._ori_documents: List[Document] = doc_loading_func(**self._yaml_config["ori_doc_loading"]["args"])

    def _save_tagged_documents(self, tagged_docs: List[Document]) -> None:
        doc_saving_module = importlib.import_module(self._yaml_config["tagged_doc_saving"]["module"])
        doc_saving_func = getattr(doc_saving_module, self._yaml_config["tagged_doc_saving"]["name"])
        doc_saving_func(tagged_docs, **self._yaml_config["tagged_doc_saving"]["args"])

    def _init_llm_client(self) -> None:
        # Dynamically import the LLM client.
        self._client_logger = Logger(name="client", dump_mode="a", dump_folder=self._yaml_config["log_dir"])

        llm_client_config = self._yaml_config["llm_client"]
        cache_location = os.path.join(
            self._yaml_config["log_dir"],
            f"{llm_client_config['cache_config']['location_prefix']}.db",
        )

        client_module = importlib.import_module(llm_client_config["module_path"])
        client_class = getattr(client_module, llm_client_config["class_name"])
        assert issubclass(client_class, BaseLLMClient)
        self._client = client_class(
            location=cache_location,
            auto_dump=llm_client_config["cache_config"]["auto_dump"],
            logger=self._client_logger,
            llm_config=llm_client_config["llm_config"],
            **llm_client_config.get("args", {}),
        )

        return

    def _init_tagger(self) -> None:
        self._init_llm_client()

        tagger_config: dict = self._yaml_config["tagger"]

        # Dynamically import the tagging communication protocol
        self._tagging_protocol = load_protocol(
            module_path=tagger_config["tagging_protocol"]["module_path"],
            protocol_name=tagger_config["tagging_protocol"]["attr_name"],
        )

        self._tag_name: str = tagger_config["tag_name"]
        self._tagger_logger = Logger(name="tagger", dump_mode="w", dump_folder=self._yaml_config["log_dir"])

        self._tagger = LLMPoweredTagger(
            llm_client=self._client,
            tagging_protocol=self._tagging_protocol,
            tag_name=self._tag_name,
            llm_config=self._yaml_config["llm_client"]["llm_config"],
            logger=self._tagger_logger,
        )

        return

    def run(self) -> None:
        tagged_docs = self._tagger.transform_documents(self._ori_documents)
        self._save_tagged_documents(tagged_docs)

        tag_num_list = [len(doc.metadata[self._tag_name]) for doc in tagged_docs]
        counter = Counter(tag_num_list)
        self._logger.info(f"{self._tag_name} counter: {counter}", tag="tagger")
