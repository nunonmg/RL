# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import os

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class OpenAIFormatHFDataset:
    """This class is used to load an SFT dataset in the OpenAI format using HuggingFace datasets.

    The dataset should be in the following format:
    {
        "conversations": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }
    
    Supports both Parquet and JSON formats automatically.
    Validation dataset is optional - if val_ds_path is None or doesn't exist, 
    only training data will be loaded.
    
    chat_key should be the key of the messages list. Multi-turn conversations are
    supported.
    The last message in the conversation must be from the assistant.
    """

    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str | None = None,
        chat_key: str = "conversations",
        system_key: str | None = None,
        system_prompt: str | None = None,
    ):
        self.chat_key = chat_key
        self.system_key = system_key
        self.system_prompt = system_prompt
        
        # Load training dataset using HuggingFace datasets
        train_original_dataset = load_dataset(train_ds_path, split="train")
        
        # Handle validation dataset - optional
        val_original_dataset = None
        if val_ds_path is not None:
            try:
                if os.path.exists(val_ds_path):
                    val_original_dataset = load_dataset(val_ds_path, split="train")
                else:
                    print(f"Warning: Validation path {val_ds_path} does not exist. Skipping validation dataset.")
            except Exception as e:
                print(f"Warning: Could not load validation dataset from {val_ds_path}: {e}")
                print("Continuing with training dataset only.")

        # Format datasets
        formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
        
        self.formatted_ds = {
            "train": formatted_train_dataset,
        }
        
        if val_original_dataset is not None:
            formatted_val_dataset = val_original_dataset.map(self.add_messages_key)
            self.formatted_ds["validation"] = formatted_val_dataset

        self.task_spec = TaskDataSpec(
            "json_dataset",
        )

    def add_messages_key(
        self,
        example: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        messages = [message for message in example[self.chat_key]]
        if self.system_key is not None and self.system_key in example:
            messages = [
                {"role": "system", "content": example[self.system_key]}
            ] + messages
        elif self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        assert messages[-1]["role"] == "assistant", f"Last message must be from assistant, got: {messages[-1]['role']}"
        return {"messages": messages}