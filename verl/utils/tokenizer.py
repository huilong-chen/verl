# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates (字节跳动有限公司及其附属公司)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Licensed under the Apache License, Version 2.0 (the "License");
# 在 Apache License 2.0 许可下获得授权
#
# you may not use this file except in compliance with the License.
# 您不得在未遵守许可证的情况下使用此文件
# You may obtain a copy of the License at
# 您可以在以下位置获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 在 "按原样" 基础上分发，不附带任何明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不附带任何形式的保证或条件，无论是明示的还是暗示的
# See the License for the specific language governing permissions and
# 请参阅许可证以了解特定语言的权限和
# limitations under the License.
# 许可证下的限制。
"""Utils for tokenization."""
"""用于分词的工具函数。"""

import warnings  # 导入警告模块，用于发出警告信息

__all__ = ["hf_tokenizer", "hf_processor"]  # 定义模块的公共接口，指定可以被导入的函数


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.
    如果 pad_token_id 为 None，则将其设置为 eos_token_id。

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.
        tokenizer (transformers.PreTrainedTokenizer): 需要设置的标记器。

    """
    # 检查标记器的 pad_token_id 是否为 None
    if tokenizer.pad_token_id is None:
        # 将 pad_token_id 设置为 eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # 发出警告，告知用户已自动设置 pad_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    # 检查标记器的 pad_token 是否为 None
    if tokenizer.pad_token is None:
        # 将 pad_token 设置为 eos_token
        tokenizer.pad_token = tokenizer.eos_token
        # 发出警告，告知用户已自动设置 pad_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.
    创建一个正确处理 eos 和 pad 标记的 huggingface 预训练标记器。

    Args:
        name (str): The name of the tokenizer.
        name (str): 标记器的名称。
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_pad_token (bool): 是否修正 pad 标记 id。
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.
        correct_gemma2 (bool): 是否修正 gemma2 标记器。

    Returns:
        transformers.PreTrainedTokenizer: The pretrained tokenizer.
        transformers.PreTrainedTokenizer: 预训练的标记器。

    """
    from transformers import AutoTokenizer  # 导入 AutoTokenizer 类用于自动加载预训练标记器

    # 检查是否需要修正 gemma2 标记器，并且名称路径是字符串且包含 "gemma-2-2b-it"
    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # gemma2 中的 EOS 标记存在歧义，可能会降低强化学习性能
        # 参考链接：https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        # 设置 eos_token 为 "<end_of_turn>"
        kwargs["eos_token"] = "<end_of_turn>"
        # 设置 eos_token_id 为 107
        kwargs["eos_token_id"] = 107
    # 从预训练模型加载标记器
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    # 如果需要修正 pad 标记，则调用 set_pad_token_id 函数
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    # 返回配置好的标记器
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.
    创建一个用于处理多模态数据的 huggingface 处理器。

    Args:
        name_or_path (str): The name of the processor.
        name_or_path (str): 处理器的名称。

    Returns:
        transformers.ProcessorMixin: The pretrained processor.
        transformers.ProcessorMixin: 预训练的处理器。
    """
    from transformers import AutoProcessor  # 导入 AutoProcessor 类用于自动加载预训练处理器

    try:
        # 尝试从预训练模型加载处理器
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        # 如果加载失败，将处理器设置为 None
        processor = None
        # TODO(haibin.lin): 在 setup.py 中添加 transformers 版本要求后应移除 try-catch 以避免静默失败
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # 避免加载标记器，参考：
    # 参考链接：https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    # 返回处理器（可能为 None）
    return processor
