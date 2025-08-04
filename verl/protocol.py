# 版权声明：本代码由字节跳动有限公司及其子公司所有
#
# 根据Apache许可证2.0版本授权使用
# 除非遵守许可证，否则不得使用此文件
# 您可以从以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 按"原样"基础分发，没有任何明示或暗示的保证或条件
# 有关许可证下权限和限制的具体语言，请参阅许可证
"""
实现任意两个函数、模块之间的基础数据传输协议。
我们可以通过继承Protocol类来定义具有特定键的更详细的批次信息。
"""

# 导入上下文管理器模块，用于忽略异常
import contextlib
# 导入深拷贝模块
import copy
# 导入日志记录模块
import logging
# 导入数学运算模块
import math
# 导入操作系统接口模块
import os
# 导入序列化模块
import pickle
# 导入数据类装饰器和字段定义
from dataclasses import dataclass, field
# 导入类型提示相关模块
from typing import Any, Callable, Optional

# 导入NumPy科学计算库
import numpy as np
# 导入Ray分布式计算框架
import ray
# 导入TensorDict库，用于处理张量字典
import tensordict
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch分布式通信模块
import torch.distributed
# 导入版本处理模块
from packaging import version
# 导入TensorDict核心类
from tensordict import TensorDict
# 导入PyTorch数据加载器
from torch.utils.data import DataLoader

# 导入设备相关工具函数
from verl.utils.device import get_device_id, get_torch_device
# 导入字典合并工具函数
from verl.utils.py_functional import union_two_dict
# 导入分布式张量收集工具函数
from verl.utils.torch_functional import allgather_dict_tensors

# 定义模块公开的API列表
__all__ = ["DataProto", "union_tensor_dict"]

# 设置TensorDict的懒加载模式，忽略可能的异常
with contextlib.suppress(Exception):
    tensordict.set_lazy_legacy(False).set()


class _DataProtoConfigMeta(type):
    """DataProto配置的元类，用于管理全局配置选项。
    
    该元类实现了自动填充功能的配置管理，支持通过环境变量
    和编程方式配置自动填充行为。
    """
    # 存储配置的字典
    _config = {}

    # 自动填充功能的配置键名
    auto_padding_key = "_verl_auto_padding"

    @property
    def auto_padding(cls):
        """获取自动填充功能的启用状态。
        
        Returns:
            bool: 如果环境变量VERL_AUTO_PADDING为TRUE/1或者显式启用，则返回True。
        """
        # 检查环境变量是否启用自动填充
        enabled_by_env = os.getenv("VERL_AUTO_PADDING", "FALSE").upper() in ["TRUE", "1"]
        # 返回环境变量或配置中设置的值
        return enabled_by_env or cls._config.get(cls.auto_padding_key, False)

    @auto_padding.setter
    def auto_padding(cls, enabled: bool):
        """设置自动填充功能的启用状态。
        
        Args:
            enabled (bool): 是否启用自动填充功能。
        """
        # 验证输入参数类型
        assert isinstance(enabled, bool), f"enabled must be a boolean, got {enabled} as {type(enabled)}"
        # 将配置存储到配置字典中
        cls._config[cls.auto_padding_key] = enabled


class DataProtoConfig(metaclass=_DataProtoConfigMeta):
    """DataProto配置类，提供全局配置选项。
    
    通过元类_DataProtoConfigMeta实现配置管理，主要提供
    自动填充功能的配置接口。
    """
    pass


# 定义填充大小的键名，用于存储数据填充的尺寸信息
_padding_size_key = "_padding_size_key_x123d"


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """将DataProto填充到能被指定除数整除的大小。

    该函数用于将DataProto的数据长度填充到指定的除数的倍数，
    这在分布式训练中很有用，可以确保数据能均匀分配。

    Args:
        data (DataProto): 需要填充的数据原型对象
        size_divisor (int): 大小除数，填充后的长度将是该值的倍数

    Returns:
        tuple: 包含两个元素的元组
            - data_padded (DataProto): 填充后的DataProto对象
            - pad_size (int): 实际填充的大小
    """
    # 验证输入数据类型
    assert isinstance(data, DataProto), "data must be a DataProto"
    # 检查当前长度是否能被除数整除
    if len(data) % size_divisor != 0:
        # 计算需要填充的大小
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        # 循环复制数据直到填充完成
        while remaining_pad > 0:
            # 确定本次复制的数量
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        # 拼接原始数据和填充数据
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        # 如果长度已经是除数的倍数，不需要填充
        if len(data) == 0:
            logging.warning("padding a DataProto with no item, no changed made")
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    """移除DataProto的填充数据。
    
    该函数用于移除之前添加的填充数据，恢复到原始大小。
    
    Args:
        data (DataProto): 包含填充数据的DataProto对象
        pad_size (int): 需要移除的填充大小
        
    Returns:
        DataProto: 移除填充后的DataProto对象
        
    Note:
        等价于执行 `data[:-pad_size]` 操作
    """
    # 如果有填充大小，则移除最后的pad_size个元素
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """合并两个TensorDict对象。
    
    该函数将第二个TensorDict中的键值对合并到第一个中，
    要求两个TensorDict具有相同的批次大小，对于相同的键
    要求张量完全相等。

    Args:
        tensor_dict1 (TensorDict): 第一个张量字典（将被修改）
        tensor_dict2 (TensorDict): 第二个张量字典（只读）

    Returns:
        TensorDict: 合并后的张量字典（实际上是tensor_dict1的引用）
    """
    # 验证两个张量字典的批次大小必须相同
    assert tensor_dict1.batch_size == tensor_dict2.batch_size, (
        f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    )
    # 遍历第二个张量字典的所有键
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            # 如果第一个字典没有该键，直接添加
            tensor_dict1[key] = tensor_dict2[key]
        else:
            # 如果第一个字典已有该键，验证两个张量是否完全相等
            assert tensor_dict1[key].equal(tensor_dict2[key]), (
                f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
            )

    return tensor_dict1


def _array_equal(array1: np.ndarray, array2: np.ndarray, visited: set[int]) -> bool:
    """
    递归比较两个NumPy数组的严格相等性，特殊处理对象类型数组、
    NaN值和循环引用。
    
    该函数假设提供的两个参数都是NumPy数组。

    Args:
        array1 (np.ndarray): 第一个NumPy数组
        array2 (np.ndarray): 第二个NumPy数组
        visited (set[int]): 已访问对象ID的集合，用于检测循环引用

    Returns:
        bool: 如果数组的数据类型、形状和所有元素都相等则返回True
    """
    # 首先检查数据类型和形状，这是最快的失败路径
    if array1.dtype != array2.dtype or array1.shape != array2.shape:
        return False

    # 对于非对象数据类型，使用NumPy的实现并设置equal_nan=True
    if array1.dtype != "object":
        return np.array_equal(array1, array2, equal_nan=True)

    # 对于对象类型数组，我们必须递归比较每个元素
    # 我们委托给_deep_equal来处理元素，因为它们可能是任何类型，
    # 包括其他嵌套数组或NaN
    return all(_deep_equal(x, y, visited) for x, y in zip(array1.flat, array2.flat, strict=False))


def _deep_equal(a: Any, b: Any, visited: set[int]) -> bool:
    """
    递归执行两个Python对象之间的深度比较。
    
    特性：
    - 正确处理NaN值（NaN == NaN 评估为True）
    - 处理循环引用
    - 如果两个对象都是NumPy数组，则分派给_array_equal
    - 否则使用标准'=='比较

    Args:
        a (Any): 第一个要比较的对象
        b (Any): 第二个要比较的对象
        visited (set[int]): 已访问对象ID的集合，用于检测循环引用

    Returns:
        bool: 如果两个对象深度相等则返回True
    """
    # 首先检查类型是否相同
    if type(a) is not type(b):
        return False

    # 如果我们在这条路径上之前见过这个对象ID，说明存在循环引用
    # 由于我们已经知道类型匹配，可以安全地假设这部分结构是相等的
    obj_id = id(a)
    if obj_id in visited:
        return True

    # 将当前对象ID添加到已访问集合中
    visited.add(obj_id)

    # 根据类型执行特定的比较
    result = False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        # 处理NaN值的特殊情况
        result = True
    elif isinstance(a, np.ndarray):
        # 我们知道b也是ndarray，因为初始类型检查已经通过
        result = _array_equal(a, b, visited)
    else:
        # 所有其他类型的标准相等性比较
        result = a == b

    # 在递归退出时清理已访问集合
    visited.remove(obj_id)
    return result


def union_numpy_dict(tensor_dict1: dict[str, np.ndarray], tensor_dict2: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """合并两个NumPy数组字典。
    
    该函数将第二个字典中的NumPy数组合并到第一个字典中，
    使用深度比较来验证相同键的数组是否完全相等。

    Args:
        tensor_dict1 (dict[str, np.ndarray]): 第一个NumPy数组字典（将被修改）
        tensor_dict2 (dict[str, np.ndarray]): 第二个NumPy数组字典（只读）

    Returns:
        dict[str, np.ndarray]: 合并后的NumPy数组字典（实际上是tensor_dict1的引用）
    """
    # 遍历第二个字典的所有键值对
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            # 验证两个值都是NumPy数组
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            # 使用深度比较来正确处理NaN和对象类型
            assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (
                f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."
            )
        # 将值赋给第一个字典（如果键已存在且验证通过，则覆盖；如果键不存在，则添加）
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """将字典列表转换为列表字典。
    
    该函数将一个字典列表转换为一个字典，其中每个键对应一个列表，
    列表中的元素是原始字典中该键对应的所有值。
    
    例如：
        [{"a": 1, "b": 2}, {"a": 3, "b": 4}] 
        -> {"a": [1, 3], "b": [2, 4]}
    
    Args:
        list_of_dict (list[dict]): 字典列表，所有字典应有相同的键集

    Returns:
        dict: 转换后的列表字典
    """
    # 如果输入列表为空，返回空字典
    if len(list_of_dict) == 0:
        return {}
    # 获取第一个字典的键作为所有字典的键集
    keys = list_of_dict[0].keys()
    # 初始化输出字典，每个键对应一个空列表
    output = {key: [] for key in keys}
    # 遍历所有字典，收集每个键的值
    for data in list_of_dict:
        for key, item in data.items():
            # 确保键存在于输出字典中
            assert key in output
            # 将值添加到对应的列表中
            output[key].append(item)
    return output


def fold_batch_dim(data: "DataProto", new_batch_size):
    """
    将批次维度从[bsz, xxx]折叠为[new_bsz, bsz // new_bsz, xxx]。
    
    该函数用于重新组织批次维度，将原来的批次大小分割为
    多个较小的批次，这在某些训练场景中很有用。

    Args:
        data (DataProto): 输入的数据原型对象
        new_batch_size (int): 新的批次大小，必须是原批次大小的因数

    Returns:
        DataProto: 重新组织后的数据原型对象
    """
    # 获取原始批次大小
    batch_size = data.batch.batch_size[0]

    # 验证新批次大小必须是原批次大小的因数
    assert batch_size % new_batch_size == 0

    # 获取张量和非张量批次数据
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch

    # 重塑张量维度：从[bsz, xxx]变为[new_bsz, bsz//new_bsz, xxx]
    tensor = tensor.view(new_batch_size, -1)
    tensor.auto_batch_size_(batch_dims=1)

    # 重塑非张量批次数据
    for key, val in non_tensor.items():
        non_tensor[key] = np.reshape(val, newshape=(new_batch_size, -1, *val.shape[1:]))

    # 返回新的DataProto对象
    return type(data)(batch=tensor, non_tensor_batch=non_tensor, meta_info=data.meta_info)


def unfold_batch_dim(data: "DataProto", batch_dims=2):
    """
    将前n个维度展开为新的批次维度。
    
    该函数执行fold_batch_dim的逆操作，将多个批次维度
    合并为一个单一的批次维度。

    Args:
        data (DataProto): 输入的数据原型对象
        batch_dims (int, optional): 要展开的批次维度数量，默认为2

    Returns:
        DataProto: 展开后的数据原型对象
    """
    # 获取张量和非张量批次数据
    tensor: TensorDict = data.batch
    non_tensor = data.non_tensor_batch
    # 设置张量的批次维度数量
    tensor.auto_batch_size_(batch_dims=batch_dims)
    # 将张量展平为一维批次
    tensor = tensor.view(-1)

    # 获取新的批次大小
    batch_size = tensor.batch_size[0]

    # 创建新的非张量批次字典
    non_tensor_new = {}

    # 重塑非张量批次数据
    for key, val in non_tensor.items():
        non_tensor_new[key] = np.reshape(val, newshape=(batch_size, *val.shape[batch_dims:]))

    # 返回新的DataProto对象
    return type(data)(batch=tensor, non_tensor_batch=non_tensor_new, meta_info=data.meta_info)


def collate_fn(x: list["DataProtoItem"]):
    """
    数据整理函数，将DataProtoItem列表整理为单个DataProto对象。
    
    该函数用于数据加载器中，将多个数据项合并为一个批次。

    Args:
        x (list[DataProtoItem]): DataProtoItem对象的列表

    Returns:
        DataProto: 合并后的DataProto对象
    """
    # 初始化批次数据列表
    batch = []
    # 初始化非张量批次数据列表
    non_tensor_batch = []
    # 收集所有数据项的张量和非张量数据
    for data in x:
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    # 将张量堆叠为一个连续的张量
    batch = torch.stack(batch).contiguous()
    # 将非张量字典列表转换为列表字典
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    # 将非张量数据转换为对象类型的NumPy数组
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.array(val, dtype=object)
    # 返回合并后的DataProto对象
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


@dataclass
class DataProtoItem:
    """DataProto项的数据类，表示单个数据项。
    
    该类表示DataProto中的一个单独数据项，包含张量数据、
    非张量数据和元信息。
    
    Attributes:
        batch (TensorDict): 张量批次数据，可以为None
        non_tensor_batch (dict): 非张量批次数据，默认为空字典
        meta_info (dict): 元信息数据，默认为空字典
    """
    # TODO(zhangchi.usc1992) 添加一致性检查
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    数据协议类，旨在为函数间的数据交换提供标准协议。
    
    DataProto是一个数据结构，包含一个批次（TensorDict）和元信息（Dict）。
    批次是一个TensorDict（https://pytorch.org/tensordict/），
    TensorDict允许您像操作单个张量一样操作张量字典。
    理想情况下，具有相同批次大小的张量应该放在批次中。
    
    Attributes:
        batch (TensorDict): 张量批次数据，存储具有相同批次大小的张量
        non_tensor_batch (dict): 非张量批次数据，存储不能或不适合作为张量的数据
        meta_info (dict): 元信息字典，存储数据的额外信息和配置
    """

    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """数据类初始化后的后处理方法。
        
        在数据类初始化完成后执行必要的检查。
        """
        # 执行必要的一致性检查
        self.check_consistency()

    def __len__(self):
        """获取DataProto的长度（批次大小）。
        
        Returns:
            int: 批次大小，如果没有数据则返回0
        """
        # 如果有张量批次，返回张量批次的第一个维度大小
        if self.batch is not None:
            return self.batch.batch_size[0]
        # 如果有非张量批次，返回第一个非张量数据的第一个维度大小
        elif self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            random_key = list(self.non_tensor_batch.keys())[0]
            return self.non_tensor_batch[random_key].shape[0]
        # 如果都没有数据，返回0
        else:
            return 0

    def __getitem__(self, item):
        """
        DataProto对象的增强索引方法。

        支持多种索引类型，提供灵活的数据访问方式。

        Args:
            item: 可以是以下类型之一：
                - int: 单个索引
                - slice: 切片对象（start:stop:step）
                - list: 索引列表
                - numpy.ndarray: 索引数组
                - torch.Tensor: 索引张量

        Returns:
            DataProto: 对于除单个整数外的所有索引类型
            DataProtoItem: 仅对单个整数索引返回
        """
        # 情况1：切片对象 - 使用slice方法
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # 情况2：列表、numpy数组或torch张量 - 使用select_idxs方法
        elif isinstance(item, list | np.ndarray | torch.Tensor):
            return self.select_idxs(item)

        # 情况3：单个整数 - 返回DataProtoItem以保持向后兼容性
        elif isinstance(item, int | np.integer):
            # 获取张量数据
            tensor_data = self.batch[item] if self.batch is not None else None
            # 获取非张量数据
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # 情况4：不支持的类型
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        """
        序列化方法，用于pickle协议。
        
        将DataProto对象转换为可序列化的格式。
        
        Returns:
            tuple: 包含序列化数据的元组
                - buffer_bytes: 张量批次的字节数据
                - non_tensor_batch: 非张量批次数据
                - meta_info: 元信息数据
        """
        import io

        # 创建字节缓冲区
        buffer = io.BytesIO()
        # 对于较新版本的TensorDict，进行内存优化
        if version.parse(tensordict.__version__) >= version.parse("0.5.0") and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        # 将张量批次保存到缓冲区
        torch.save(self.batch, buffer)
        # 获取字节数据
        buffer_bytes = buffer.getvalue()
        return buffer_bytes, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        """
        反序列化方法，用于pickle协议。
        
        从序列化数据恢复DataProto对象。
        
        Args:
            data (tuple): 序列化的数据元组
        """
        import io

        # 解包序列化数据
        batch_deserialized_bytes, non_tensor_batch, meta_info = data
        # 从字节数据创建字节流
        batch_deserialized = io.BytesIO(initial_bytes=batch_deserialized_bytes)
        # 加载张量批次，根据设备可用性选择映射位置
        batch = torch.load(
            batch_deserialized,
            weights_only=False,
            map_location="cpu" if not get_torch_device().is_available() else None,
        )
        # 恢复对象属性
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def save_to_disk(self, filepath):
        """
        将DataProto对象保存到磁盘文件。
        
        Args:
            filepath (str): 保存文件的路径
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_disk(filepath) -> "DataProto":
        """
        从磁盘文件加载DataProto对象。
        
        Args:
            filepath (str): 要加载的文件路径
            
        Returns:
            DataProto: 加载的DataProto对象
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            return data

    def print_size(self, prefix=""):
        """
        打印DataProto对象的内存使用情况。
        
        计算并打印张量批次和非张量批次的内存占用（以GB为单位）。
        
        Args:
            prefix (str, optional): 打印消息的前缀，默认为空字符串
        """
        # 初始化张量字典大小
        size_of_tensordict = 0
        # 计算张量批次的总大小
        if self.batch is not None:
            for _, tensor in self.batch.items():
                size_of_tensordict += tensor.element_size() * tensor.numel()
        # 初始化NumPy数组大小
        size_of_numpy_array = 0
        # 计算非张量批次的总大小
        for _, numpy_array in self.non_tensor_batch.items():
            size_of_numpy_array += numpy_array.nbytes

        # 转换为GB单位
        size_of_numpy_array /= 1024**3
        size_of_tensordict /= 1024**3

        # 构建打印消息
        message = f"Size of tensordict: {size_of_tensordict} GB, size of non_tensor_batch: {size_of_numpy_array} GB"

        # 如果有前缀，添加到消息中
        if prefix:
            message = f"{prefix}, " + message
        print(message)

    def check_consistency(self):
        """
        检查DataProto的一致性，主要针对批次和非张量批次。
        
        该函数作为公共方法暴露，用户可以直接调用以确保数据的一致性。
        验证批次维度、数据类型和批次大小的一致性。
        """
        # 检查张量批次的一致性
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        # 检查非张量批次的数据类型
        if self.non_tensor_batch is not None:
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray)

        # 当同时存在张量批次和非张量批次时的额外检查
        if self.batch is not None and self.non_tensor_batch is not None and len(self.non_tensor_batch) != 0:
            # TODO: 如果需要，我们实际上可以放宽这个限制
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            # 获取批次大小
            batch_size = self.batch.batch_size[0]
            # 检查每个非张量数据的一致性
            for key, val in self.non_tensor_batch.items():
                assert isinstance(val, np.ndarray), (
                    f"data in the non_tensor_batch must be a numpy.array with dtype=object, but for "
                    f"{key=}, got {type(val)=}"
                )
                # 检查批次大小是否一致
                assert val.shape[0] == batch_size, (
                    f"key {key} length {len(val)} is not equal to batch size {batch_size}"
                )

    @classmethod
    def from_single_dict(cls, data: dict[str, torch.Tensor | np.ndarray], meta_info=None, auto_padding=False):
        """从张量和非张量字典创建DataProto。
        
        该方法接受一个包含张量和NumPy数组的字典，自动分类并创建DataProto对象。
        
        Args:
            data (dict[str, torch.Tensor | np.ndarray]): 包含张量和NumPy数组的字典
            meta_info (dict, optional): 元信息字典，默认为None
            auto_padding (bool, optional): 是否启用自动填充，默认为False
            
        Returns:
            DataProto: 创建的DataProto对象
            
        Raises:
            ValueError: 当字典中包含不支持的数据类型时抛出
        """
        # 初始化张量和非张量字典
        tensors = {}
        non_tensors = {}

        # 遍历输入字典，按类型分类
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        # 调用from_dict方法创建DataProto对象
        return cls.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info, auto_padding=auto_padding)

    @classmethod
    def from_dict(
        cls,
        tensors: Optional[dict[str, torch.Tensor]] = None,
        non_tensors=None,
        meta_info=None,
        num_batch_dims=1,
        auto_padding=False,
    ):
        """从张量字典创建DataProto。
        
        该方法从张量字典创建DataProto对象，假设：
        1. 张量字典中的所有张量在第0维有相同的大小
        2. 只有第0维是批次维度
        
        Args:
            tensors (Optional[dict[str, torch.Tensor]]): 张量字典，默认为None
            non_tensors (Optional[dict]): 非张量字典，默认为None
            meta_info (Optional[dict]): 元信息字典，默认为None
            num_batch_dims (int, optional): 批次维度数量，默认为1
            auto_padding (bool, optional): 是否启用自动填充，默认为False
            
        Returns:
            DataProto: 创建的DataProto对象
        """
        # 验证批次维度数量必须大于0
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        # 如果有非张量数据，批次维度数量必须为1
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        # 设置默认值
        if tensors is None:
            tensors = {}
        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        # 验证非张量参数类型
        assert isinstance(non_tensors, dict)

        # 获取并检查批次大小
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                # 第一个张量，设置批次大小基准
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                # 后续张量，验证批次大小是否一致
                current_batch = tensor.shape[:num_batch_dims]
                assert batch_size == current_batch, (
                    f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. "
                    f"Got {pivot_key} has {batch_size}, {key} has {current_batch}"
                )

        # 确保非张量数据都是NumPy数组
        for key, val in non_tensors.items():
            if not isinstance(val, np.ndarray):
                non_tensors[key] = np.array(val, dtype=object)

        # 创建TensorDict对象
        tensor_dict = TensorDict(source=tensors, batch_size=batch_size) if tensors else None
        # 如果启用自动填充，设置相应的元信息
        if auto_padding:
            meta_info[DataProtoConfig.auto_padding_key] = True
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """将批次数据移动到指定设备。

        将张量批次数据移动到指定的设备（如GPU或CPU）。
        
        Args:
            device (torch.device, str): PyTorch设备对象或设备字符串

        Returns:
            DataProto: 当前DataProto对象（支持链式调用）
        """
        # 如果有张量批次，将其移动到指定设备
        if self.batch is not None:
            self.batch = self.batch.to(device)
        return self

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """选择DataProto的子集。

        通过指定的键选择批次数据、非张量批次数据和元信息的子集。
        
        Args:
            batch_keys (list, optional): 要选择的批次键列表
            non_tensor_batch_keys (list, optional): 要选择的非张量批次键列表
            meta_info_keys (list, optional): 要选择的元信息键列表
            deepcopy (bool, optional): 是否进行深拷贝，默认为False

        Returns:
            DataProto: 包含选定键的DataProto对象
        """
        # TODO (zhangchi.usc1992) 是否需要复制
        # 选择张量批次子集
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        # 选择非张量批次子集
        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        # 如果需要，进行深拷贝
        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        # 选择元信息子集
        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        # 如果需要，进行深拷贝
        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        # 返回新的DataProto对象
        return type(self)(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        从DataProto中选择特定的索引。

        根据提供的索引选择数据，支持多种索引类型。

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): 要选择的索引
                可以是布尔掩码、整数索引或索引列表

        Returns:
            DataProto: 包含选定索引的新DataProto对象
        """
        # 处理列表类型的索引
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        # 处理NumPy数组类型的索引
        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        # 计算选择后的批次大小
        batch_size = int(idxs_np.sum()) if idxs_np.dtype == bool else idxs_np.shape[0]

        # 选择张量批次数据
        if self.batch is not None:
            # 使用TensorDict的内置索引功能
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()},
                batch_size=(batch_size,),
                device=self.batch.device,
            )
        else:
            selected_batch = None

        # 选择非张量批次数据
        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        # 返回新的DataProto对象
        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        对DataProto进行切片并返回新的DataProto对象。
        
        这是直接切片的改进版本，直接切片返回DataProtoItem，
        而此方法返回DataProto。

        Args:
            start (int, optional): 起始索引，默认为None（从开头开始）
            end (int, optional): 结束索引（不包含），默认为None（到结尾）
            step (int, optional): 步长，默认为None（步长为1）

        Returns:
            DataProto: 包含切片数据的新DataProto对象

        Examples:
            # 直接使用slice方法
            sliced_data = data_proto.slice(10, 20)

            # 使用增强索引（返回DataProto）
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # 每隔一个元素

            # 使用列表索引（返回DataProto）
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # 单个索引仍然返回DataProtoItem
            single_item = data_proto[5]
        """
        # 创建切片对象
        slice_obj = slice(start, end, step)

        # 处理批次数据
        if self.batch is not None:
            # 使用TensorDict的内置切片功能
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # 处理非张量批次数据
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # 返回新的DataProto对象
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """从DataProto中弹出一个子集。
        
        通过指定的键从DataProto中移除并返回对应的数据。
        
        Args:
            batch_keys (list, optional): 要弹出的批次键列表
            non_tensor_batch_keys (list, optional): 要弹出的非张量批次键列表
            meta_info_keys (list, optional): 要弹出的元信息键列表

        Returns:
            DataProto: 包含被弹出数据的新DataProto对象
        """
        # 设置默认值
        if batch_keys is None:
            batch_keys = []
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []

        # 初始化弹出数据字典
        tensors = {}
        # 弹出张量批次数据
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # 弹出非张量批次数据
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        # 弹出元信息数据
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        # 从弹出的数据创建新的DataProto对象
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        重命名批次数据中的键。
        
        注意：此函数仅重命名批次中的键，不影响非张量批次和元信息。
        
        Args:
            old_keys (str or list): 原键名或键名列表
            new_keys (str or list): 新键名或键名列表

        Returns:
            DataProto: 当前DataProto对象（支持链式调用）
        """
        def validate_input(keys):
            """验证输入键参数的有效性。"""
            if keys is not None:
                if isinstance(keys, str):
                    keys = [keys]
                elif isinstance(keys, list):
                    pass
                else:
                    raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
            return keys

        # 验证输入参数
        old_keys = validate_input(old_keys)
        new_keys = validate_input(new_keys)

        # 检查新旧键的数量是否匹配
        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        # 执行重命名操作
        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """与另一个DataProto进行合并。
        
        分别合并批次数据、非张量批次数据和元信息。
        在以下情况下会抛出错误：
        - 批次数据中存在冲突的键且它们不相等
        - 两个数据批次的批次大小不同
        - 元信息中存在冲突的键且它们不相同

        Args:
            other (DataProto): 要合并的另一个DataProto对象

        Returns:
            DataProto: 合并后的DataProto对象
        """
        # 合并张量批次数据
        self.batch = union_tensor_dict(self.batch, other.batch)
        # 合并非张量批次数据
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        # 合并元信息数据
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        r"""从DataProto创建迭代器。
        
        该功能基于TensorDict可以作为普通PyTorch数据集使用的事实。
        更多详情请参见：https://pytorch.org/tensordict/tutorials/data_fashion

        Args:
            mini_batch_size (int): 迭代数据集时的小批次大小。要求
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): 迭代数据集的轮数
            dataloader_kwargs (Any): 内部返回一个DataLoader来处理批次。
                dataloader_kwargs是传递给DataLoader的参数

        Returns:
            Iterator: 每次生成一个小批次数据的迭代器。总迭代步数为
                ``self.batch.batch_size * epochs // mini_batch_size``
        """
        # 验证批次大小能被小批次大小整除
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # 我们可以直接从TensorDict创建数据加载器
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        # 设置随机种子生成器
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        # 验证数据加载器参数类型
        assert isinstance(dataloader_kwargs, dict)
        # 创建训练数据加载器
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            """数据生成器函数。"""
            for _ in range(epochs):
                for d in train_dataloader:
                    # 保持元信息
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def is_padding_enabled(self):
        """
        检查DataProto是否启用了填充功能。
        
        Returns:
            bool: 如果启用填充功能则返回True，否则返回False
        """
        # 检查DataProto特定的填充设置
        dataproto_specific_padding = self.meta_info.get(DataProtoConfig.auto_padding_key, False)
        # 返回特定设置或全局设置中的任何一个为True
        return dataproto_specific_padding or DataProtoConfig.auto_padding

    def padding(self, padding_size, padding_candidate=""):
        """通过连接重复的填充候选项来填充DataProto。

        Args:
            padding_size (int): 重复填充候选项的次数
            padding_candidate (str): 要重复并附加到DataProto的候选项，
                仅支持["first", "last"]，默认为空字符串（使用最后一个）
        """
        # 如果填充大小为0，直接返回
        if padding_size == 0:
            return
        # 选择填充候选项：第一个或最后一个元素
        padding_candidate = self.select_idxs([0 if padding_candidate == "first" else len(self) - 1])
        # 重复填充候选项
        padding_part = padding_candidate.repeat(padding_size)
        # 将原始数据与填充数据连接
        padded_dp = DataProto.concat([self, padding_part])
        # 更新当前对象的批次数据
        self.batch = padded_dp.batch
        self.non_tensor_batch = padded_dp.non_tensor_batch

    def chunk(self, chunks: int) -> list["DataProto"]:
        """将批次沿第0维分割成多个块。
        
        元信息会传递给每个分割后的DataProto。

        Args:
            chunks (int): 要在第0维上分割的块数

        Returns:
            List[DataProto]: 分割后的DataProto列表
        """
        # 如果未启用填充，检查长度是否能被块数整除
        if not self.is_padding_enabled():
            assert len(self) % chunks == 0, (
                f"only support equal chunk. Got size of DataProto {len(self)} and chunk {chunks}."
            )

        bsz_in_batch = None
        # 分割张量批次数据
        if self.batch is not None:
            batch_lst = self.batch.chunk(chunks=chunks, dim=0)
            bsz_in_batch = np.array([batch.batch_size[0] for batch in batch_lst])
            chunk_indices = np.cumsum(bsz_in_batch)[:-1]
        else:
            batch_lst = [None for _ in range(chunks)]

        # 分割非张量批次数据
        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            if bsz_in_batch is not None:
                # 使用累积索引进行分割
                non_tensor_lst = np.array_split(val, chunk_indices.tolist())
            else:
                # 简单等分
                non_tensor_lst = np.array_split(val, chunks)
            assert len(non_tensor_lst) == chunks
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        # 构建输出列表
        output = []
        for i in range(chunks):
            output.append(
                type(self)(batch=batch_lst[i], non_tensor_batch=non_tensor_batch_lst[i], meta_info=self.meta_info)
            )

        return output

    def split(self, split_size: int) -> list["DataProto"]:
        """将批次沿第0维按指定大小分割。
        
        元信息会传递给每个分割后的DataProto。

        Args:
            split_size (int): 每个分割的大小

        Returns:
            List[DataProto]: 分割后的DataProto列表
        """
        return [self[i : i + split_size] for i in range(0, len(self), split_size)]

    @staticmethod
    def concat(data: list["DataProto"]) -> "DataProto":
        """连接DataProto列表。
        
        批次数据沿第0维连接。假设元信息相同，将使用第一个的元信息。

        Args:
            data (List[DataProto]): DataProto列表

        Returns:
            DataProto: 连接后的DataProto
        """
        # 收集所有批次数据
        batch_lst = []
        for batch in data:
            batch_lst.append(batch.batch)
        # 连接张量批次数据
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst[0] is not None else None

        # 连接非张量批次数据
        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        # 确定返回类型并创建结果对象
        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=data[0].meta_info)

    def reorder(self, indices):
        """
        根据索引重新排序数据。
        
        注意：此操作是原地操作，会修改当前对象。
        
        Args:
            indices (torch.Tensor): 重新排序的索引
        """
        # 将索引转换为NumPy数组
        indices_np = indices.detach().numpy()
        # 重新排序张量批次数据
        self.batch = self.batch[indices]
        # 重新排序非张量批次数据
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def repeat(self, repeat_times=2, interleave=True):
        """
        重复批次数据指定的次数。

        Args:
            repeat_times (int): 重复数据的次数，默认为2
            interleave (bool): 是否交错重复的数据，默认为True

        Returns:
            DataProto: 包含重复数据的新DataProto对象
        """
        # 处理张量批次数据的重复
        if self.batch is not None:
            if interleave:
                # 交错重复数据
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # 堆叠数据
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            # 创建重复后的张量字典
            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        # 处理非张量批次数据的重复
        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                # 交错重复
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                # 平铺重复
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        # 返回新的DataProto对象
        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def unfold_column_chunks(self, n_split: int, split_keys: Optional[list[str]] = None):
        """沿第二维分割成`n_split`，然后展开到第一维（批次维度）。
        
        在传递不希望在数据集中被洗牌的分组张量时很有用。
        不在split_keys中的键会被重复以匹配形状。
        注意：如果未提供`split_keys`，它将重复第二维中的所有键。
        
        Args:
            n_split (int): 分割的数量
            split_keys (Optional[list[str]]): 要分割的键列表，默认为None

        Returns:
            DataProto: 展开后的DataProto对象
        """
        # 处理张量批次数据的展开
        if self.batch is not None:
            unfolded_batch = {}
            for key in self.batch.keys():
                if key in split_keys if split_keys is not None else False:
                    # 对指定的键进行分割和展开
                    shape = list(self.batch[key].shape)
                    shape[0] = self.batch[key].shape[0] * n_split
                    shape[1] = self.batch[key].shape[1] // n_split
                    unfolded_batch[key] = self.batch[key].reshape(*shape)
                else:
                    # 对其他键进行重复
                    unfolded_batch[key] = torch.repeat_interleave(self.batch[key], n_split, dim=0)
            # 将展开的批次定位为与原始批次相同设备上的TensorDict
            unfolded_batch = TensorDict(
                source=unfolded_batch, batch_size=(self.batch.batch_size[0] * n_split,), device=self.batch.device
            )
        else:
            unfolded_batch = None

        # 处理非张量批次数据的重复
        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if key in split_keys:
                # 对指定的键进行分割和展开
                shape = list(val.shape)
                shape[0] = val.shape[0] * n_split
                shape[1] = val.shape[1] // n_split
                repeated_non_tensor_batch[key] = val.reshape(*shape)
            else:
                # 对其他键进行重复
                repeated_non_tensor_batch[key] = np.repeat(val, n_split, axis=0)

        return type(self)(
            batch=unfolded_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def sample_level_repeat(self, repeat_times):
        """
        按样本级别重复批次数据中的每一行指定的次数。
        
        每行可以有不同的重复次数，这在处理不平衡数据时很有用。

        Args:
            repeat_times (torch.tensor, list, tuple, ndarray): 每行重复的次数

        Returns:
            DataProto: 包含重复数据的新DataProto对象
        """
        # 标准化repeat_times为列表格式
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        elif isinstance(repeat_times, torch.Tensor):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        elif isinstance(repeat_times, np.ndarray):
            assert len(repeat_times.shape) == 1
            repeat_times = repeat_times.tolist()
        else:
            assert isinstance(repeat_times, list), (
                f"repeat_times type must be in [list, torch.Tensor, np.ndarray, tuple], got {type(repeat_times)}"
            )
        # 转换为PyTorch张量
        repeat_times = torch.tensor(repeat_times)

        # 处理张量批次数据的重复
        if self.batch is not None:
            # 交错重复数据
            repeated_tensors = {
                key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
            }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(repeat_times.sum().item(),),
                device=self.batch.device,
            )
        else:
            repeated_batch = None

        # 处理非张量批次数据的重复
        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    def get_data_info(self) -> str:
        """返回存储数据的格式化信息，包含嵌套类型详细信息。

        Returns:
            str: 显示张量详细信息和递归元数据类型的格式化字符串
        """
        info = ["batch"]

        # 添加张量批次信息
        for key, tensor in self.batch.items():
            if hasattr(tensor, "shape") and hasattr(tensor, "dtype") and hasattr(tensor, "device"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype}) {tensor.device}")
            elif hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
                info.append(f"  {key}: {tuple(tensor.shape)} ({tensor.dtype})")
            else:
                info.append(f"  {key}: {type(tensor).__name__}")

        # 添加非张量批次信息
        info.append("non_tensor_batch")
        for key, array in self.non_tensor_batch.items():
            info.append(f"  {key}: ndarray{array.shape} ({array.dtype})")

        # 添加元信息信息
        info.append("meta_info")
        for k, v in self.meta_info.items():
            type_info = self._get_type_info(v)
            info.append(f"  {k}: {type_info}")

        return "\n".join(info)

    def _get_type_info(self, value):
        """递归获取嵌套结构的类型信息。
        
        Args:
            value: 要分析类型的值
            
        Returns:
            str: 类型的字符串表示
        """
        if isinstance(value, list):
            # 处理列表类型，分析前3个元素的类型
            elem_types = {self._get_type_info(v) for v in value[:3]}
            return f"list[{'|'.join(elem_types) if elem_types else '...'}]"
        if isinstance(value, tuple):
            # 处理元组类型，分析所有元素的类型
            elem_types = [self._get_type_info(v) for v in value]
            return f"tuple({', '.join(elem_types)})"
        if isinstance(value, dict):
            # 处理字典类型，分析键值对的类型
            if not value:
                return "dict"
            k, v = next(iter(value.items()))
            return f"dict[{self._get_type_info(k)}: {self._get_type_info(v)}]"
        if isinstance(value, np.ndarray):
            # 处理NumPy数组类型
            return f"ndarray{value.shape} ({value.dtype})"
        # 处理基本类型
        return type(value).__name__


@dataclass
class DataProtoFuture:
    """
    DataProtoFuture类，旨在消除驱动程序上的实际数据获取。
    
    通过这种方式，驱动程序不必等待数据，从而实现异步执行。
    DataProtoFuture包含来自另一个WorkerGroup的大小为world_size的future列表。
    - collect_fn是一个可调用对象，将future列表减少为单个DataProto
    - dispatch_fn是一个可调用对象，将DataProto分区为大小为world_size的DataProto列表，然后进行选择

    潜在问题：我们可以优化dispatch_fn(collect_fn)，使得只在目标位置获取需要的数据
    - DataProtoFuture只支持直接从方法的输出传递到另一个输入。您不能在驱动程序中对DataProtoFuture执行任何操作。
    
    Attributes:
        collect_fn (Callable): 收集函数，用于将future列表合并为DataProto
        futures (list[ray.ObjectRef]): Ray对象引用列表
        dispatch_fn (Callable, optional): 分发函数，默认为None
    """

    collect_fn: Callable
    futures: list[ray.ObjectRef]
    dispatch_fn: Callable = None

    @staticmethod
    def concat(data: list[ray.ObjectRef]) -> "DataProtoFuture":
        """
        创建用于连接操作的DataProtoFuture。
        
        Args:
            data (list[ray.ObjectRef]): 要连接的Ray对象引用列表
            
        Returns:
            DataProtoFuture: 用于连接操作的future对象
        """
        output = DataProtoFuture(collect_fn=DataProto.concat, futures=data)
        return output

    def chunk(self, chunks: int) -> list["DataProtoFuture"]:
        """
        将DataProtoFuture分割成多个块。
        
        Args:
            chunks (int): 要分割的块数
            
        Returns:
            list[DataProtoFuture]: 分割后的DataProtoFuture列表
        """
        from functools import partial

        arg_future_lst = []
        for i in range(chunks):
            # 注意我们不能直接传递i和chunks参数
            def dispatch_fn(x, i, chunks):
                return x.chunk(chunks=chunks)[i]

            arg_future = DataProtoFuture(
                collect_fn=self.collect_fn, dispatch_fn=partial(dispatch_fn, i=i, chunks=chunks), futures=self.futures
            )
            arg_future_lst.append(arg_future)
        return arg_future_lst

    def get(self):
        """
        获取DataProtoFuture的实际数据。
        
        Returns:
            DataProto: 获取并处理后的数据
        """
        # 获取future的实际数据
        output = ray.get(self.futures)  # 数据并行大小
        # 验证所有输出都是DataProto类型
        for o in output:
            assert isinstance(o, DataProto)
        # 使用收集函数处理数据
        output = self.collect_fn(output)  # 选择数据并行，连接数据
        # 如果有分发函数，使用它进一步处理数据
        if self.dispatch_fn is not None:
            output = self.dispatch_fn(output)  # 在批次维度分割，使用数据并行选择
        return output


def all_gather_data_proto(data: DataProto, process_group):
    """
    在分布式环境中收集所有进程的DataProto数据。
    
    注意：这是一个原地操作，就像torch.distributed.all_gather一样。
    
    Args:
        data (DataProto): 要收集的DataProto数据
        process_group: 进程组对象
    """
    # 获取进程组的大小
    group_size = torch.distributed.get_world_size(group=process_group)
    # 验证输入数据类型
    assert isinstance(data, DataProto)
    # 保存原始设备信息
    prev_device = data.batch.device
    # 将张量批次移动到当前设备
    data.batch = data.batch.to(get_device_id())
    # 执行张量字典的allgather操作
    data.batch = allgather_dict_tensors(data.batch.contiguous(), size=group_size, group=process_group, dim=0)
    # 将张量批次移回原始设备
    data.batch = data.batch.to(prev_device)
    # 收集非张量批次数据
    all_non_tensor_batch = [None for _ in range(group_size)]
    torch.distributed.all_gather_object(all_non_tensor_batch, data.non_tensor_batch, group=process_group)
    # 合并所有进程的非张量批次数据
    data.non_tensor_batch = {k: np.concatenate([d[k] for d in all_non_tensor_batch]) for k in data.non_tensor_batch}
