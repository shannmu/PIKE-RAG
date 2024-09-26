# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import Any


def parse_json(content: str) -> Any:
    content = content.replace("\n", " ")
    start_idx = content.find("{")
    end_idx = content.find("}")
    content = content[start_idx : end_idx + 1]
    return json.loads(content)


def parse_json_v2(content: str) -> Any:
    content = content.replace("\n", " ")

    start_idx = content.rfind(': "')
    end_idx = content.rfind('"}')
    if start_idx >= 0 and end_idx >= 0:
        content = content[:start_idx] + ': "' + content[start_idx + len(': "') : end_idx].replace('"', "") + '"}'

    start_idx = content.find("{")
    end_idx = content.find("}")
    content = content[start_idx : end_idx + 1]
    return json.loads(content, strict=False)
