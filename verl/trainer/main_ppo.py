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
注意：我们不将main函数与ray_trainer合并，因为ray_trainer被其他main函数使用
"""

# 导入操作系统相关模块
import os
# 导入网络通信相关模块
import socket

# 导入Hydra配置管理框架
import hydra
# 导入Ray分布式计算框架
import ray
# 导入OmegaConf配置对象处理库
from omegaconf import OmegaConf

# 导入抽象采样器基类
from verl.experimental.dataset.sampler import AbstractSampler
# 导入PPO训练的Ray运行时环境配置
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
# 导入PPO训练器
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# 导入奖励管理器
from verl.trainer.ppo.reward import load_reward_manager
# 导入CUDA设备可用性检查函数
from verl.utils.device import is_cuda_available
# 导入外部类型加载工具
from verl.utils.import_utils import load_extern_type


# 使用Hydra装饰器定义主函数入口，配置文件路径为config/ppo_trainer.yaml
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """PPO训练的主入口函数，使用Hydra进行配置管理。

    Args:
        config: Hydra配置对象，包含所有训练参数。
    """
    # 调用PPO训练执行函数
    run_ppo(config)


# 定义PPO训练流程执行函数
def run_ppo(config) -> None:
    """初始化Ray集群并运行分布式PPO训练过程。

    Args:
        config: 训练配置对象，包含分布式PPO训练所需的所有参数，
                包括Ray初始化设置、模型路径和训练超参数。
    """
    # 检查Ray是否未初始化
    if not ray.is_initialized():
        # 使用本地集群配置初始化Ray
        # 在运行时环境中设置环境变量，用于控制分词器并行度、
        # NCCL调试级别、VLLM日志级别，并允许运行时LoRA更新
        # `num_cpus`指定Ray可使用的CPU核心数，从配置中获取
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    # 创建TaskRunner类的远程实例，并
    # 远程执行TaskRunner实例的`run`方法并等待完成
    if (
        is_cuda_available  # 检查CUDA是否可用
        and config.trainer.get("profile_steps") is not None  # 检查是否配置了性能分析步骤
        and len(config.trainer.get("profile_steps", [])) > 0  # 检查是否有性能分析步骤
    ):
        # 导入NVTX可用性检查函数
        from verl.utils.import_utils import is_nvtx_available

        # 确保NVTX在CUDA平台上可用
        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        # 将配置转换为字典格式
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        # 创建带有Nsight性能分析选项的远程TaskRunner实例
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        # 创建普通的远程TaskRunner实例
        runner = TaskRunner.remote()
    # 远程执行TaskRunner的run方法并等待结果
    ray.get(runner.run.remote(config))

    # [可选] 从配置中获取时间线跟踪文件路径，默认为None
    # 该文件用于性能分析
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        # 生成时间线跟踪文件
        ray.timeline(filename=timeline_json_file)


# 使用Ray远程装饰器定义任务运行器类，分配1个CPU核心
# 请确保主任务不在头节点上调度
@ray.remote(num_cpus=1)
class TaskRunner:
    """用于执行分布式PPO训练任务的Ray远程类。

    该类封装了主要的训练逻辑，作为Ray远程actor运行，
    以实现跨多个节点和GPU的分布式执行。
    """

    def run(self, config):
        """执行主要的PPO训练工作流程。

        此方法设置分布式训练环境，初始化
        工作器、数据集和奖励函数，然后启动训练过程。

        Args:
            config: 训练配置对象，包含设置和运行PPO训练过程
                   所需的所有参数。
        """
        # 打印初始配置，`resolve=True`将评估符号值
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # 打印当前任务运行器的主机名和进程ID
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # 打印解析后的配置内容
        pprint(OmegaConf.to_container(config, resolve=True))
        # 解析配置中的符号值
        OmegaConf.resolve(config)

        # 从HDFS下载检查点到本地机器
        # `use_shm`决定是否使用共享内存，开启可能导致模型加载更快
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # 实例化分词器和处理器
        from verl.utils import hf_processor, hf_tokenizer

        # 获取是否信任远程代码的配置
        trust_remote_code = config.data.get("trust_remote_code", False)
        # 创建分词器实例
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # 创建处理器实例（用于多模态LLM，可能为None）
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 根据actor策略定义工作器类
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            # 确保critic策略与actor策略一致
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            # 获取是否使用传统工作器实现的配置
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # import warnings
                # warnings.warn(f"Legacy worker impl is going to be deprecated, will be removed in the future. \
                #   Please set trainer.use_legacy_worker_impl = false to switch to the new worker implementation.")
                # 导入传统CriticWorker
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                # 导入新实现的CriticWorker
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            # 根据rollout模式选择工作器类
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            # 设置Ray工作器组类
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # Megatron策略下确保actor和critic策略一致
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # 根据rollout模式选择工作器类
            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            # 设置Megatron Ray工作器组类
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 将角色映射到对应的远程工作器类
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # 定义资源池规范
        # 将角色映射到资源池
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # 我们应该采用多源奖励函数：
        # - 对于基于规则的奖励模型，我们直接调用奖励分数
        # - 对于基于模型的奖励模型，我们调用模型
        # - 对于代码相关的提示，如果有测试用例，我们发送到沙箱
        # 最后，我们将所有奖励组合在一起
        # 奖励类型取决于数据的标签
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            # 添加奖励模型工作器到角色映射
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 如果使用KL损失或KL奖励，添加参考策略工作器
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 加载用于训练和验证的奖励管理器
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        # 创建资源池管理器
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # 创建训练和验证数据集
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        # 创建训练采样器
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 初始化PPO训练器
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        # 初始化训练器的工作器
        trainer.init_workers()
        # 启动训练过程
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """创建强化学习数据集。

    Arguments:
        data_paths: 数据文件路径列表。
        data_config: 数据配置对象。
        tokenizer (Tokenizer): 分词器实例。
        processor (Processor): 处理器实例。

    Returns:
        dataset (Dataset): 创建的数据集实例。
    """
    # 导入PyTorch数据集基类
    from torch.utils.data import Dataset

    # 导入默认的RLHF数据集类
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 检查数据配置中是否指定了自定义数据集类
    # 以及是否提供了自定义类的路径
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # 动态加载自定义数据集类
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # 验证自定义数据集类是否继承自torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"来自'{data_config.custom_cls.path}'的自定义数据集类"
                f"'{data_config.custom_cls.name}'必须继承自torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # 如果指定了数据生成策略，使用DynamicGenDataset类
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # 如果没有指定自定义类，使用默认的RLHFDataset类
        dataset_cls = RLHFDataset
    # 打印使用的数据集类名
    print(f"Using dataset class: {dataset_cls.__name__}")

    # 使用确定的数据集类实例化数据集
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """为数据集创建采样器。

    Arguments:
        data_config: 数据配置对象。
        dataset (Dataset): 数据集实例。

    Returns:
        sampler (Sampler): 创建的采样器实例。
    """
    # 导入PyTorch相关模块
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # 检查是否配置了自定义采样器类
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        # 动态加载课程学习采样器类
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        # 创建采样器实例
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        # 验证采样器是否为AbstractSampler的实例
        assert isinstance(sampler, AbstractSampler)
        # 确保使用课程学习时num_workers为0，防止数据缓存
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "如果使用课程学习，num_workers必须为0以防止数据缓存。"
            "如果数据加载器在批次完成前缓存数据，"
            "课程采样器将无法重新排序数据。"
        )

    # 使用采样器来便于检查点恢复。
    # 如果在数据配置中启用了洗牌，则创建随机采样器。
    elif data_config.shuffle:
        # 创建训练数据加载器的随机数生成器
        train_dataloader_generator = torch.Generator()
        # 设置随机种子以确保可重复性
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        # 创建随机采样器
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # 如果禁用洗牌，使用顺序采样器按顺序遍历数据集。
        sampler = SequentialSampler(data_source=dataset)

    return sampler


# 程序入口点，当直接运行此脚本时执行主函数
if __name__ == "__main__":
    main()
