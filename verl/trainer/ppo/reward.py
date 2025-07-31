# Copyright 2025 Individual Contributor: Thibaut Barroyer
# Copyright 2025 个人贡献者：Thibaut Barroyer
#
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

import multiprocessing  # 导入多进程模块，用于并行处理
import os  # 导入操作系统接口模块，用于文件系统操作
from functools import partial  # 导入 partial 函数，用于函数部分参数绑定

import ray  # 导入 Ray 分布式计算框架

from verl import DataProto  # 导入 VERL 框架的数据协议类
from verl.utils.reward_score import default_compute_score  # 导入默认奖励计算函数


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.
    通过将 `extra_kwargs` 合并到调用时的 `kwargs` 中来调用 `raw_fn`，其中 `extra_kwargs` 具有优先权。

    This function is used to merge additional keyword arguments with the original function's arguments.
    此函数用于将额外的关键字参数与原始函数的参数合并。
    """
    # 合并关键字参数，extra_kwargs 的优先级高于 kwargs
    merged_kwargs = {**kwargs, **extra_kwargs}
    # 调用原始函数并传入合并后的参数
    return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config):
    """Load and return a custom reward function from external file.
    从外部文件加载并返回自定义奖励函数。

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.
    从指定文件路径动态导入奖励函数，并用配置中的额外关键字参数包装它。

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
        config (dict): 包含 custom_reward_function 设置的配置字典
                      settings with 'path', 'name', and 'reward_kwargs' fields.
                      包含 'path'、'name' 和 'reward_kwargs' 字段的设置。

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
        callable 或 None: 包装了合并 kwargs 的奖励函数，如果没有配置
                         if no custom reward function is configured.
                         自定义奖励函数则返回 None。

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        FileNotFoundError: 如果指定的奖励函数文件不存在。
        RuntimeError: If there's an error loading the module from file.
        RuntimeError: 如果从文件加载模块时出错。
        AttributeError: If the specified function name isn't found in the module.
        AttributeError: 如果在模块中找不到指定的函数名。
    """
    import importlib.util  # 导入模块导入工具
    import sys  # 导入系统模块

    # 获取自定义奖励函数配置，如果不存在则使用空字典
    reward_fn_config = config.get("custom_reward_function") or {}
    # 获取文件路径
    file_path = reward_fn_config.get("path")
    # 如果没有提供文件路径，返回 None
    if not file_path:
        return None

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    # 从文件位置创建模块规范
    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    # 根据规范创建模块对象
    module = importlib.util.module_from_spec(spec)
    try:
        # 将模块添加到 sys.modules 中
        sys.modules["custom_module"] = module
        # 执行模块代码
        spec.loader.exec_module(module)
    except Exception as e:
        # 如果加载失败，抛出运行时错误
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    # 获取函数名
    function_name = reward_fn_config.get("name")
    # 检查模块中是否存在该函数
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    # 打印使用自定义奖励函数的信息
    print(f"using customized reward function '{function_name}' from '{file_path}'")
    # 获取原始函数对象
    raw_fn = getattr(module, function_name)

    # 获取奖励函数的关键字参数，如果不存在则使用空字典
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    # 返回包装后的函数，将额外参数合并到函数调用中
    return partial(_call_with_kwargs, raw_fn, reward_kwargs)


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """
    Load and initialize a reward manager based on the configuration.
    根据配置加载并初始化奖励管理器。

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        config: 包含 reward_model 字段的 PPO 训练器配置对象。
        tokenizer: Tokenizer object used for processing text.
        tokenizer: 用于处理文本的标记器对象。
        num_examine: Number of samples to examine.
        num_examine: 要检查的样本数量。
        **reward_kwargs: Additional keyword arguments for the reward manager.
        **reward_kwargs: 奖励管理器的额外关键字参数。

    Returns:
        An instance of the specified reward manager class.
        指定奖励管理器类的实例。
    """
    from verl.workers.reward_manager import get_reward_manager_cls  # 导入奖励管理器类获取函数

    # 预定义的奖励管理器列表在 `verl/workers/reward_manager/` 中定义：
    # naive: NaiveRewardManager - 朴素奖励管理器
    # prime: PrimeRewardManager - 主要奖励管理器
    # batch: BatchRewardManager - 批量奖励管理器
    # dapo: DAPORewardManager - DAPO 奖励管理器
    # Note(haibin.lin): 对于自定义奖励管理器，请确保它们已通过 `verl.workers.reward_manager.register` 导入并注册
    # 默认情况下 reward_manager 设置为 naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    # 根据名称获取奖励管理器类
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # 尝试根据配置获取自定义奖励函数
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    # 如果没有自定义奖励函数，则使用默认或沙箱配置
    if compute_score is None:
        # 获取沙箱融合配置
        sandbox_config = config.reward_model.get("sandbox_fusion")
        # 获取沙箱 URL，如果配置不存在则为 None
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        # 获取内存限制，默认为 1024 MB
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
        # 如果配置了沙箱 URL
        if sandbox_url:
            # 创建多进程管理器
            sandbox_manager = multiprocessing.Manager()
            # 创建信号量来控制对沙箱的并发访问
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            # 使用沙箱配置包装默认计算函数
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            # 使用默认的奖励计算函数
            final_compute_score = default_compute_score

    # 使用指定参数实例化并返回奖励管理器
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    计算一批数据的奖励。

    Args:
        data: DataProto object containing the input data.
        data: 包含输入数据的 DataProto 对象。
        reward_fn: Reward function to compute the reward.
        reward_fn: 用于计算奖励的奖励函数。

    Returns:
        Tuple of reward tensor and extra info dictionary.
        奖励张量和额外信息字典的元组。
    """
    try:
        # 尝试调用奖励函数并返回字典格式结果
        reward_result = reward_fn(data, return_dict=True)
        # 从结果中提取奖励张量
        reward_tensor = reward_result["reward_tensor"]
        # 从结果中提取额外信息，如果不存在则使用空字典
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        # 如果调用失败，打印错误信息
        print(f"Error in reward_fn: {e}")
        # 回退到直接调用奖励函数
        reward_tensor = reward_fn(data)
        # 设置额外的信息字典为空
        reward_extra_infos_dict = {}

    # 返回奖励张量和额外信息字典
    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)  # Ray 远程装饰器，指定每个工作进程使用 1 个 CPU
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    加载奖励管理器并计算一批数据的奖励。
    
    This is meant to be run in a separate Ray worker.
    这意味着在单独的 Ray 工作进程中运行。
    """
    # 如果没有提供奖励函数
    if reward_fn is None:
        # 断言配置和标记器不能为 None
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )
        import warnings  # 导入警告模块

        # 发出弃用警告，建议使用 reward_fn 参数而不是 config 和 tokenizer
        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        # 加载奖励管理器作为奖励函数
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    # 调用同步奖励计算函数并返回结果
    return compute_reward(data, reward_fn)
