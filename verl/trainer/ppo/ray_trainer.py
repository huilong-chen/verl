# 版权所有 2024 Bytedance Ltd. 和/或其附属公司
# 版权所有 2023-2024 SGLang Team
# 版权所有 2025 ModelBest Inc. 和/或其附属公司
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 按"原样"分发，不附带任何明示或暗示的保证或条件。
# 请参阅许可证以了解有关权限和限制的具体语言。
"""
基于 Ray 单控制器的 PPO 训练器。
此训练器支持使用 huggingface 进行模型不可知的模型初始化
"""

# 导入 JSON 处理库，用于数据的序列化和反序列化
import json
# 导入操作系统接口模块，用于文件和目录操作
import os
# 导入 UUID 生成库，用于生成唯一标识符
import uuid
# 导入警告模块，用于控制警告信息的显示
import warnings
# 导入默认字典类，用于创建具有默认值的字典
from collections import defaultdict
# 导入深拷贝函数，用于创建对象的完整副本
from copy import deepcopy
# 导入数据类装饰器和字段函数，用于简化类的定义
from dataclasses import dataclass, field
# 导入枚举类，用于创建枚举类型
from enum import Enum
# 导入美化打印函数，用于格式化输出复杂数据结构
from pprint import pprint
# 导入可选类型注解，用于类型提示
from typing import Optional

# 导入 NumPy 数值计算库，用于高效的数组操作
import numpy as np
# 导入 Ray 分布式计算框架，用于构建分布式应用
import ray
# 导入 PyTorch 深度学习框架，用于张量计算和神经网络
import torch
# 导入 OmegaConf 配置管理库，用于处理配置文件
from omegaconf import OmegaConf, open_dict
# 导入 PyTorch 数据集和数据采样器
from torch.utils.data import Dataset, Sampler
# 导入状态化数据加载器，用于保持数据加载状态
from torchdata.stateful_dataloader import StatefulDataLoader
# 导入进度条库，用于显示训练进度
from tqdm import tqdm

# 导入 VERL 框架的核心数据协议
from verl import DataProto
# 导入实验性的课程学习采样器
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
# 导入数据协议的填充和取消填充工具函数
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
# 导入基础工作器类
from verl.single_controller.base import Worker
# 导入 Ray 相关的类和资源池
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
# 导入 Ray 基础模块中的工作器创建函数
from verl.single_controller.ray.base import create_colocated_worker_cls
# 导入算法配置类
from verl.trainer.config import AlgoConfig
# 导入 PPO 核心算法模块
from verl.trainer.ppo import core_algos
# 导入优势估计器和损失聚合函数
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
# 导入指标计算工具函数
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,          # 计算数据相关指标
    compute_throughout_metrics,    # 计算吞吐量指标
    compute_timing_metrics,        # 计算时间相关指标
    process_validation_metrics,    # 处理验证指标
)
# 导入奖励计算函数（同步和异步）
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
# 导入检查点管理工具函数
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
# 导入配置转换工具函数
from verl.utils.config import omega_conf_to_dataclass
# 导入调试计时器工具
from verl.utils.debug import marked_timer
# 导入指标聚合工具函数
from verl.utils.metric import reduce_metrics
# 导入序列长度平衡工具函数
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
# 导入 PyTorch 函数式工具中的掩码均值计算
from verl.utils.torch_functional import masked_mean
# 导入验证生成日志记录器
from verl.utils.tracking import ValidationGenerationsLogger

# 定义工作器类型为工作器类的类型别名
WorkerType = type[Worker]


class Role(Enum):
    """
    训练系统中不同角色的枚举定义。
    
    要动态创建更多角色，您可以继承 Role 并添加新成员。
    每个角色代表训练过程中的一个特定功能组件。
    """

    # Actor 角色：负责策略网络的前向传播和参数更新
    Actor = 0
    # Rollout 角色：负责生成经验轨迹和环境交互
    Rollout = 1
    # ActorRollout 角色：结合 Actor 和 Rollout 功能的混合角色
    ActorRollout = 2
    # Critic 角色：负责价值函数的估计和更新
    Critic = 3
    # RefPolicy 角色：参考策略，用于 KL 散度计算和策略约束
    RefPolicy = 4
    # RewardModel 角色：奖励模型，用于评估动作的质量
    RewardModel = 5
    # ActorRolloutRef 角色：结合 Actor、Rollout 和 Ref 功能的混合角色
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    资源池管理器：定义和管理分布式训练中的资源池规范。
    
    资源池将在其他组件之前首先初始化，确保资源分配的合理性。
    负责检查资源可用性并创建相应的 Ray 资源池。
    """

    # 资源池规范字典：键为资源池名称，值为每个节点上的进程数量列表
    resource_pool_spec: dict[str, list[int]]
    # 角色到资源池名称的映射字典
    mapping: dict[Role, str]
    # 资源池字典：存储已创建的 Ray 资源池实例
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """为分布式训练创建 Ray 资源池。

        根据资源池规范初始化资源池，每个池管理跨多个节点的 GPU 资源。
        对于 FSDP 后端，使用 max_colocate_count=1 来合并 WorkerGroups。
        对于 Megatron 后端，使用 max_colocate_count>1 来支持不同模型。
        """
        # 遍历资源池规范中的每个资源池配置
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count 表示每个 RayResourcePool 中的 WorkerGroups（即进程）数量
            # 对于 FSDP 后端，建议使用 max_colocate_count=1 将所有 WorkerGroups 合并为一个
            # 对于 Megatron 后端，建议使用 max_colocate_count>1
            # 这样可以为不同模型利用不同的 WorkerGroup
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,    # 每个节点上的进程数量
                use_gpu=True,                       # 启用 GPU 使用
                max_colocate_count=1,              # 最大共定位计数
                name_prefix=resource_pool_name     # 资源池名称前缀
            )
            # 将创建的资源池存储到字典中
            self.resource_pool_dict[resource_pool_name] = resource_pool

        # 检查资源是否可用
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """获取指定角色对应的资源池。
        
        Args:
            role (Role): 要获取资源池的角色类型
            
        Returns:
            RayResourcePool: 对应角色的 Ray 资源池实例
        """
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """获取集群中 GPU 的总数。
        
        Returns:
            int: 集群中可用的 GPU 总数
        """
        # 计算所有资源池中 GPU 数量的总和
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """检查 Ray 集群中的资源是否满足资源池的需求。
        
        该方法验证集群是否有足够的 GPU/NPU 资源来满足所有资源池的要求。
        如果资源不足，将抛出 ValueError 异常。
        """
        # 获取每个节点的可用资源信息
        node_available_resources = ray.state.available_resources_per_node()
        # 提取每个节点的可用 GPU/NPU 数量
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # 检查所需的总 GPU 数量是否可以得到满足
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"可用的 GPU 总数 {total_available_gpus} 小于所需的 GPU 总数 {total_required_gpus}"
            )

        # 检查每个资源池是否可以得到满足，时间复杂度为 O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            # 遍历每个节点，尝试分配所需的 GPU 资源
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    # 从该节点分配所需的 GPU 数量
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            # 如果还有未满足的节点需求，抛出异常
            if num_nodes > 0:
                raise ValueError(
                    f"资源池 {resource_pool_name}: {num_gpus}*{num_nodes} "
                    + "无法在此 Ray 集群中得到满足"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """对 token 级别的奖励应用 KL 散度惩罚。

    该函数计算参考策略和当前策略之间的 KL 散度，
    然后基于这个散度对 token 级别的奖励应用惩罚。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据。
        kl_ctrl (core_algos.AdaptiveKLController): 自适应 KL 惩罚控制器。
        kl_penalty (str, optional): 要应用的 KL 惩罚类型。默认为 "kl"。

    Returns:
        tuple: 包含以下内容的元组：
            - 更新后的数据，其中 token 级别奖励已根据 KL 惩罚进行调整
            - 与 KL 惩罚相关的指标字典
    """
    # 获取响应掩码，用于标识哪些位置是响应部分
    response_mask = data.batch["response_mask"]
    # 获取 token 级别的分数（原始奖励）
    token_level_scores = data.batch["token_level_scores"]
    # 获取批次大小
    batch_size = data.batch.batch_size[0]

    # 计算参考策略和当前策略之间的 KL 散度
    # 当应用 KL 惩罚时，algorithm.use_kl_in_reward=True，所以参考模型已被启用
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # 输出形状: (batch_size, response_length)
    # 将 KL 散度乘以响应掩码，只对响应部分应用惩罚
    kld = kld * response_mask
    # 获取当前的 KL 惩罚系数
    beta = kl_ctrl.value

    # 计算 token 级别的最终奖励：原始奖励减去 KL 惩罚
    token_level_rewards = token_level_scores - beta * kld

    # 计算当前的平均 KL 散度（在序列维度上平均）
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # 在序列上求平均
    current_kl = torch.mean(current_kl, dim=0).item()  # 在批次上求平均并转换为标量

    # 参考：https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    # 更新 KL 控制器
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    # 将计算得到的 token 级别奖励保存回数据中
    data.batch["token_level_rewards"] = token_level_rewards

    # 构建与 KL 惩罚相关的指标字典
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """计算序列中响应部分的注意力掩码。

    该函数提取与模型响应相对应的注意力掩码部分，
    该掩码用于屏蔽只应应用于响应 token 的计算。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据。

    Returns:
        torch.Tensor: 响应 token 的注意力掩码。
    """
    # 获取响应序列
    responses = data.batch["responses"]
    # 获取响应序列的长度
    response_length = responses.size(1)
    # 获取完整的注意力掩码
    attention_mask = data.batch["attention_mask"]
    # 返回注意力掩码的最后 response_length 列，即响应部分的掩码
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """计算用于策略优化的优势估计。

    该函数使用各种估计器（如 GAE、GRPO、REINFORCE++ 等）计算优势估计。
    优势估计用于指导强化学习算法中的策略优化。

    Args:
        data (DataProto): 包含批处理模型输出和输入的数据。
        adv_estimator (AdvantageEstimator): 要使用的优势估计器（例如：GAE、GRPO、REINFORCE++）。
        gamma (float, optional): 未来奖励的折扣因子。默认为 1.0。
        lam (float, optional): GAE 的 Lambda 参数。默认为 1.0。
        num_repeat (int, optional): 重复计算的次数。默认为 1。
        norm_adv_by_std_in_grpo (bool, optional): 是否在 GRPO 中按标准差归一化优势。
            默认为 True。
        config (dict, optional): 算法设置的配置字典。默认为 None。

    Returns:
        DataProto: 更新后的数据，包含计算得到的优势和回报。
    """
    # 与不在 fit 中计算响应掩码的训练器向后兼容
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    
    # 根据不同的优势估计器类型进行计算
    if adv_estimator == AdvantageEstimator.GAE:
        # 使用广义优势估计（GAE）计算优势和回报
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],  # token 级别奖励
            values=data.batch["values"],                         # 价值函数估计
            response_mask=data.batch["response_mask"],           # 响应掩码
            gamma=gamma,                                           # 折扣因子
            lam=lam,                                               # GAE lambda 参数
        )
        # 将计算得到的优势和回报保存到数据中
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        # 如果使用 PF-PPO，进行数据重新加权
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),    # 重新加权方法
                config.pf_ppo.get("weight_pow"),          # 权重幂次
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # 初始化 GRPO 计算的掩码
        grpo_calculation_mask = data.batch["response_mask"]
        # 调用 GRPO 结果优势计算函数，参数匹配其定义
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],  # token 级别奖励
            response_mask=grpo_calculation_mask,                    # GRPO 计算掩码
            index=data.non_tensor_batch["uid"],                    # 样本索引
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,        # 是否按标准差归一化
        )
        # 将计算得到的优势和回报保存到数据中
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # 处理除 GAE 和 GRPO 之外的所有其他优势估计器类型
        # 获取对应的优势估计器函数
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        # 构建优势估计器的参数字典
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        # 可选参数：如果有 uid，添加到参数中
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        # 可选参数：如果有奖励基线，添加到参数中
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # 计算优势估计器
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        # 将计算得到的优势和回报保存到数据中
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    
    return data


class RayPPOTrainer:
    """使用 Ray 进行可扩展强化学习的分布式 PPO 训练器。

    该训练器在多个节点和 GPU 上协调分布式 PPO 训练，
    使用 Ray 后端管理 actor rollouts、critic 训练和奖励计算。
    支持多种模型架构，包括 FSDP、Megatron 和 vLLM 集成。
    """

    # TODO: 支持每个角色有独立的 ray_worker_group_cls，
    # 即支持不同角色的不同后端
    def __init__(
        self,
        config,                                        # 训练配置对象
        tokenizer,                                    # 分词器，用于文本编码和解码
        role_worker_mapping: dict[Role, WorkerType],   # 角色到工作器类的映射
        resource_pool_manager: ResourcePoolManager,    # 资源池管理器
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,  # Ray 工作器组类
        processor=None,                               # 可选的数据处理器，用于多模态数据
        reward_fn=None,                               # 训练期间计算奖励的函数
        val_reward_fn=None,                           # 验证期间计算奖励的函数
        train_dataset: Optional[Dataset] = None,      # 训练数据集
        val_dataset: Optional[Dataset] = None,        # 验证数据集
        collate_fn=None,                              # 将数据样本整理成批次的函数
        train_sampler: Optional[Sampler] = None,      # 训练数据集的采样器
        device_name=None,                             # 训练设备名称（如 "cuda", "cpu"）
    ):
        """
        使用 Ray 后端初始化分布式 PPO 训练器。
        注意：此训练器在单个 CPU/GPU 节点的驱动进程上运行。

        Args:
            config: 包含训练参数的配置对象。
            tokenizer: 用于编码和解码文本的分词器。
            role_worker_mapping (dict[Role, WorkerType]): 从角色到工作器类的映射。
            resource_pool_manager (ResourcePoolManager): Ray 资源池的管理器。
            ray_worker_group_cls (RayWorkerGroup, optional): Ray 工作器组的类。默认为 RayWorkerGroup。
            processor: 可选的数据处理器，用于多模态数据。
            reward_fn: 训练期间计算奖励的函数。
            val_reward_fn: 验证期间计算奖励的函数。
            train_dataset (Optional[Dataset], optional): 训练数据集。默认为 None。
            val_dataset (Optional[Dataset], optional): 验证数据集。默认为 None。
            collate_fn: 将数据样本整理成批次的函数。
            train_sampler (Optional[Sampler], optional): 训练数据集的采样器。默认为 None。
            device_name (str, optional): 训练设备名称（例如："cuda", "cpu"）。默认为 None。
        """

        # 存储用于文本处理的分词器
        self.tokenizer = tokenizer
        # 存储多模态数据处理器
        self.processor = processor
        # 存储训练配置
        self.config = config
        # 存储训练奖励函数
        self.reward_fn = reward_fn
        # 存储验证奖励函数
        self.val_reward_fn = val_reward_fn

        # 检查是否使用混合引擎
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "目前只支持混合引擎"

        # 如果使用混合引擎，确保 ActorRollout 角色存在
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        # 存储角色到工作器的映射
        self.role_worker_mapping = role_worker_mapping
        # 存储资源池管理器
        self.resource_pool_manager = resource_pool_manager
        # 检查是否使用参考策略
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        # 检查是否使用奖励模型
        self.use_rm = Role.RewardModel in role_worker_mapping
        # 存储 Ray 工作器组类
        self.ray_worker_group_cls = ray_worker_group_cls
        # 设置设备名称
        self.device_name = device_name if device_name else self.config.trainer.device
        # 初始化验证生成日志记录器
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,      # 项目名称
            experiment_name=self.config.trainer.experiment_name, # 实验名称
        )

        # 如果 ref_in_actor 为 True，参考策略将是没有应用 LoRA 的 actor
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # 定义奖励中的 KL 控制
        # KL 损失控制目前不支持
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # 检查是否使用 critic
        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            # 如果使用 GAE 优势估计器，则需要 critic
            self.use_critic = True
        else:
            # 如果不是 GAE 且未明确启用 critic，发出警告
            warnings.warn(
                "由于算法优势估计器不是 GAE，已禁用 critic。"
                "如果这不是预期的，请设置 critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        # 验证配置并创建数据加载器
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        """验证训练配置的正确性和一致性。
        
        该方法检查各种配置参数的有效性，确保它们相互兼容且满足训练要求。
        包括 GPU 数量、批次大小、微批次设置等多个方面的验证。
        """
        config = self.config
        # 计算总 GPU 数量：每节点 GPU 数量 × 节点数
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        
        # 如果使用 Megatron 策略，进行特定的配置验证
        if config.actor_rollout_ref.actor.strategy == "megatron":
            # 计算 Megatron 模型并行大小
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size     # 张量模型并行大小
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size  # 流水线模型并行大小
            )
            # 验证 GPU 总数能被模型并行大小和上下文并行大小的乘积整除
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"GPU 总数 ({n_gpus}) 必须能被模型并行大小 ({model_parallel_size}) 乘以 "
                f"上下文并行大小 ({config.actor_rollout_ref.actor.megatron.context_parallel_size}) 整除"
            )
            # 计算 Megatron 数据并行大小
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            # 计算最小的批次大小
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            # 非 Megatron 策略下，最小批次大小就是 GPU 总数
            minimal_bsz = n_gpus

        # 1. 检查总批次大小以确保数据正确性
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"实际训练批次大小 ({real_train_batch_size}) 必须能被最小可能批次大小 "
            f"({minimal_bsz}) 整除"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """使用 Ray 后端初始化分布式训练工作器。

        该方法负责创建和初始化所有必要的分布式组件：
        1. 根据配置创建 Ray 资源池
        2. 为每个角色（actor、critic 等）创建工作器组
        3. 初始化异步 rollout 管理器（如果配置了异步模式）
        """
        # 创建资源池
        self.resource_pool_manager.create_resource_pool()

        # 初始化资源池到类的映射字典
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        """保存训练检查点。
        
        该方法保存当前训练状态到本地和可选的远程存储，
        包括 actor 模型、critic 模型和数据加载器状态。
        """
        from verl.utils.fs import local_mkdir_safe

        # 构建检查点路径：给定路径 + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"本地全局步数文件夹: {local_global_step_folder}")
        # 构建 actor 模型的本地保存路径
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        # 构建 actor 模型的远程保存路径（如果配置了 HDFS）
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        PPO 的训练循环主函数。
        
        驱动进程只需要通过 RPC 调用工作器组的计算函数来构建 PPO 数据流。
        轻量级的优势计算在驱动进程上完成。
        这是整个训练过程的核心入口点。
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        # 初始化日志记录器，用于记录训练过程中的各种指标
        logger = Tracking(
            project_name=self.config.trainer.project_name,          # 项目名称
            experiment_name=self.config.trainer.experiment_name,   # 实验名称
            default_backend=self.config.trainer.logger,           # 默认日志后端
            config=OmegaConf.to_container(self.config, resolve=True),  # 配置信息
        )

        # 初始化全局步数计数器
        self.global_steps = 0

        # 在进行任何操作之前加载检查点（如果存在）
        self._load_checkpoint()

        # 在训练开始之前执行验证
        # 目前，我们只支持使用 reward_function 进行验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()           # 执行验证
            assert val_metrics, f"{val_metrics=}"   # 确保验证指标不为空
            pprint(f"初始验证指标: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)  # 记录验证指标
            # 如果只进行验证，则直接返回
            if self.config.trainer.get("val_only", False):
                return

        # 添加 tqdm 进度条以可视化训练进度
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="训练进度")

        # 我们从第 1 步开始
        self.global_steps += 1
        # 存储最后一次验证指标
        last_val_metrics = None
        # 记录最大步骤持续时间，用于性能监控
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        # 遍历所有训练轮次
        for epoch in range(self.config.trainer.total_epochs):
            # 遍历训练数据加载器中的每个批次
            for batch_dict in self.train_dataloader:
                # 初始化指标字典和时间记录字典
                metrics = {}
                timing_raw = {}

                # 使用计时器标记性能分析开始
                with marked_timer("start_profile", timing_raw):
                    # 根据配置决定是否开始性能分析
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                # 将批次字典转换为 DataProto 对象
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 移除用于生成的键，准备生成批次
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                # 根据批次内容动态添加需要移除的键
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                # 从批次中移除指定的键，创建生成批次
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # 将全局步数传递给生成批次用于追踪
                gen_batch.meta_info["global_steps"] = self.global_steps
                # 重复生成批次以进行多次 rollout（通常用于提高样本多样性）
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                # 检查是否为最后一步
                is_last_step = self.global_steps >= self.total_training_steps

                # 使用计时器标记整个训练步骤
                with marked_timer("step", timing_raw):
                    # ========== 生成批次 ==========
                    with marked_timer("gen", timing_raw, color="red"):
                        # 根据是否使用异步 rollout 模式选择生成方式
                        if not self.async_rollout_mode:
                            # 同步模式：直接使用工作器组生成序列
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            # 异步模式：使用异步 rollout 管理器生成序列
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        # 更新时间统计信息
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        # 移除时间信息以避免数据冗余
                        gen_batch_output.meta_info.pop("timing", None)

                    # ========== REMAX 优势估计的特殊处理 ==========
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        # REMAX 需要奖励函数来计算基线
                        if self.reward_fn is None:
                            raise ValueError("REMAX 优势估计需要奖励函数。")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            # 创建基线生成批次（不进行采样，使用贪婪解码）
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            # 生成基线序列
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            # 将基线输出合并到主批次
                            batch = batch.union(gen_baseline_output)
                            # 计算基线奖励
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            # 移除基线批次的键，只保留基线奖励
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            # 存储奖励基线张量
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            # 清理内存
                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ========== 奖励计算 ==========
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # 计算奖励模型分数
                        if self.use_rm:
                            # 如果使用奖励模型，通过工作器组计算分数
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # 根据配置选择同步或异步奖励计算
                        if self.config.reward_model.launch_reward_fn_async:
                            # 异步模式：启动远程奖励计算任务
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            # 同步模式：直接计算奖励
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # ========== 重新计算旧的对数概率 ==========
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        # 计算当前策略下的对数概率
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # 提取熵值和响应掩码
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        # 获取损失聚合模式
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # 计算熵的聚合值
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        # 构建熵指标
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        # 移除熵信息以避免数据冗余
                        old_log_prob.batch.pop("entropys")
                        # 将对数概率信息合并到批次中
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # ========== 优势计算 ==========
                    with marked_timer("adv", timing_raw, color="brown"):
                        # 我们结合基于规则的奖励模型
                        reward_extra_infos_dict: dict[str, list]
                        # 如果使用异步奖励计算，获取异步结果
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # 将奖励张量存储到批次中
                        batch.batch["token_level_scores"] = reward_tensor

                        # 如果有额外的奖励信息，将其添加到批次中
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # 计算最终奖励。如果可用，应用 KL 惩罚
                        if self.config.algorithm.use_kl_in_reward:
                            # 应用 KL 惩罚到奖励中
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            # 不使用 KL 惩罚，直接使用原始奖励
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # 计算优势估计，在驱动进程上执行
                        # 获取 GRPO 优势归一化因子
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO 优势归一化因子

                        # 调用优势计算函数
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,      # 优势估计器类型
                            gamma=self.config.algorithm.gamma,                     # 折扣因子
                            lam=self.config.algorithm.lam,                         # GAE lambda 参数
                            num_repeat=self.config.actor_rollout_ref.rollout.n,    # 重复次数
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,      # GRPO 归一化设置
                            config=self.config.algorithm,                         # 算法配置
                        )

                    # ========== 更新 Critic ==========
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            # 通过 critic 工作器组更新 critic 网络
                            critic_output = self.critic_wg.update_critic(batch)
                        # 聚合 critic 输出的指标
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # ========== 实现 Critic 预热 ==========
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # ========== 更新 Actor ==========
                        with marked_timer("update_actor", timing_raw, color="red"):
                            # 设置多轮对话信息
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # 通过 actor 工作器组更新 actor 网络
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # 聚合 actor 输出的指标
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # ========== 训练指标收集 ==========
                metrics.update(
                    {
                        "training/global_step": self.global_steps,  # 全局步数
                        "training/epoch": epoch,                    # 当前轮次
                    }
                )
                # 收集数据相关指标
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # 收集时间相关指标
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: 实现实际的 TFLOP 和理论 TFLOP 计算
                # 计算吞吐量指标
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # 这是实验性功能，未来可能会被更改或删除，以支持通用功能
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    # 如果使用课程学习采样器，更新采样器状态
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: 创建支持各种后端的规范日志记录器
                # 记录所有指标到日志系统
                logger.log(data=metrics, step=self.global_steps)

                # 更新进度条
                progress_bar.update(1)
                # 增加全局步数
                self.global_steps += 1

                # 检查是否为最后一步，如果是则结束训练
                if is_last_step:
                    pprint(f"最终验证指标: {last_val_metrics}")
                    progress_bar.close()
                    return

                # 这是实验性功能，未来可能会被更改或删除
                # 以支持通用数据缓冲池
                if hasattr(self.train_dataset, "on_batch_end"):
                    # 数据集可能在每个训练批次后发生变化
                    self.train_dataset.on_batch_end(batch=batch)
