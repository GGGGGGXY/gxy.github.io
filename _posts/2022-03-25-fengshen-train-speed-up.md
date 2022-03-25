---
title: 如何给你的大模型训练加速
tags: ['大模型训练', 'NLP', '机器学习']
---


随着预训练语言模型的提出，自然语言处理进入了全新的时代。在更丰富的数据集，参数量更大的模型，不仅仅是在各种下游任务上取得了SOTA的效果，更是在一些任务上已经过超过人类。然而随着模型变大，机器资源在一定程度上却没有跟上，在训练、微调的过程中也存在诸多问题，大多数使用者并没有庞大的计算资源，那么如何在有限的资源上，让你的模型训练尽可能的快呢，就让本文带你进入大模型训练的世界。

本文将以Finetune[**封神榜开源模型-二郎神**](https://huggingface.co/IDEA-CCNL/Erlangshen-1.3B)为例，如何快速Finetune这个13亿参数的模型(1.3B parameters = 24 layers, 2048 hidden size, 8 attention heads)

- 下游任务：[AFQMC](https://github.com/CLUEbenchmark/CLUEPretrainedModels/)
- framework: pytorch-lightning v1.6.0 dev
- device: NVIDIA DGX(A100 40G * 8)

在本篇文章中，仅仅尝试使用了一些模型训练加速的方法去提升我们的训练效率，在后续的文章中我们会详细的对每种方法进行细致的剖析，挖了一个大坑，敬请期待。

### Finetune代码

完整的代码篇幅比较长，就不在这里全部贴出来了，完整的代码可以在我们的完整的代码篇幅比较长，就不在这里全部贴出来了，完整的代码可以在我们的[**封神榜-LM**](https://github.com/IDEA-CCNL/Fengshenbang-LM)项目上找到。

finetune_classification.py, 这里可以简单看看我们的main函数。

```jsx
def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--pretrained_model_path', default='', type=str)
    total_parser.add_argument('--output_save_path',
                              default='./predict.json', type=str)

    # * Args for data preprocessing
    total_parser = TaskDataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = pl.Trainer.add_argparse_args(total_parser)
    total_parser = TaskModelCheckpoint.add_argparse_args(total_parser)

    # * Args for base model
    total_parser = LitAutoEncoder.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    checkpoint_callback = TaskModelCheckpoint(args).callbacks
    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback]
                                            )
		# 所有的数据处理流程
    data_model = TaskDataModel(args)
		# 模型，封装了所有的训练逻辑在这里面
    model = LitAutoEncoder(args, len(data_model.train_dataloader()))

		# 利用pytorch-lightning.Trainer进行训练
    trainer.fit(model, data_model)
    result = trainer.predict(model, data_model)
    save_test(result, args, data_model)
```

然后就是选取合适的batch_size就可以开始训练啦，怎么估算多少batch_size可以刚好把咱们的显存占满呢，这里可以通过公式估算一下，（total_gpu_mem - model_used_mem）// activation_mem

```jsx
// 参数占用显存大小
params_mem = num_params * (2 if fp16_enabled else 4)
// 梯度占用显存大小
gradients_mem = num_params * (2 if fp16_enabled else 4)
// 优化器占用显存大小
optimizer_mem = num_params * (16 if fp16_enabled else 8)
```

13亿参数的模型，加载好以后如果开始训练，差不多需要26G，通过上面的公式估算而来，在用剩下的14G显存除以激活值占用的显存，激活值占用的显存可以用模型经过一次前向后，前后的显存差值来获得。最终我们的batch_size选取在4。

### 开始训练

直接通过脚本finetune_classification_els_afqms.sh就可以开始训练，脚本也可以在我们的github下找到，我们的参数设置如下

```jsx
TASK=afqmc
TEXTA_NAME=sentence1
TEXTB_NAME=sentence2
LABEL_NAME=label
ID_NAME=id

BATCH_SIZE=4
VAL_BATCH_SIZE=32

# afqmc的数据路径，修改成你的路径
DATA_DIR=?
PRETRAINED_MODEL_PATH='IDEA-CCNL/Erlangshen-1.3B'

# 模型保存的一些参数
CHECKPOINT_PATH=?
DEFAULT_ROOT_DIR=?
OUTPUT_PATH=?

# 数据相关的参数
DATA_ARGS="\
        --data_dir $DATA_DIR \
        --train_data train.json \
        --valid_data dev.json \
        --test_data test.json \
        --train_batchsize $BATCH_SIZE \
        --valid_batchsize $VAL_BATCH_SIZE \
        --max_length 512 \
        --texta_name $TEXTA_NAME \
        --textb_name $TEXTB_NAME \
        --label_name $LABEL_NAME \
        --id_name $ID_NAME \
        "
# 模型相关参数
MODEL_ARGS="\
        --learning_rate 1e-5 \
        --weight_decay 1e-1 \
        --warmup 0.01 \
        --num_labels 2 \
        "

# CKPT保存相关参数
MODEL_CHECKPOINT_ARGS="\
        --monitor val_acc \
        --save_top_k 3 \
        --mode max \
        --every_n_train_steps 200 \
        --save_weights_only True \
        --dirpath $CHECKPOINT_PATH \
        --filename model-{epoch:02d}-{val_acc:.4f} \
        "

# 这里设置一共跑5个epoch，使用1 gpu
TRAINER_ARGS="\
        --max_epochs 5 \
        --gpus 1 \
        --gradient_clip_val 1.0 \
        --val_check_interval 1.0 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "

options=" \
        --pretrained_model_path $PRETRAINED_MODEL_PATH \
        --output_save_path $OUTPUT_PATH \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

SINGULARITY_PATH=/cognitive_comp/gaoxinyu/docker/pytorch21_06_py3_docker_image_v2.sif
SCRIPT_PATH=../../examples/finetune_classification.py

CUDA_VISIBLE_DEVICES=5 singularity exec --nv -B /cognitive_comp/:/cognitive_comp/ $SINGULARITY_PATH python3 $SCRIPT_PATH $options
# if you don't have docker, you can use the following command
# python3 $SCRIPT_PATH $options
```

运行上面的脚本轻松就可以起下游任务，lightning会输出我们的任务需要跑多久

```jsx
Epoch 0:  12% 1121/9664 [07:18<55:44,  2.55it/s, loss=1.44, v_num=8]
```

可以看到上面07:18<55:44，左边是已经运行的时间，右边是剩余的时间，特意让模型已经跑了一段时间，计算时间也相对比较稳定，一个epoch需要9664个step，需要63分钟，差不多一个小时，这还只是一个epoch，5个就需要五个小时，并且这里我使用的是40G的显卡，如果batch_size小的话，需要的时间更久，那么有没有优化空间呢？

这里我们分成两种情况：

- 我有多张显卡
- 我只有一张显卡

### 多张显卡

当我们有多张显卡的时候，我们可以使用多张显卡进行训练，这里就涉及到了并行的概念。目前我们常用的技术有:

- 数据并行
- 模型并行
- 流水线并行

数据并行通常用于模型加速训练，后两种通常用于训练更大的模型。因为我们这里一台机器有8张GPU，所以可以尝试使用数据并行，这里简单介绍一些数据并行，在后续文章中，我们也会给大家详细介绍一下各种并行技术的前生今世。

数据并行会自动切割我们的数据，并将不同的数据作业(batch)提交到不同的GPU上，在不同的GPU上进行forward backward，每个GPU完成任务后，汇总数据后再开始下一轮的训练，相当于提高了我们的batch_size，提高了训练的效率。

数据并行通常又分为**DataParallel**和**DistributedDataParallel**，简称DP和DDP，这两种区别也会我们后面系列的文章中做详细的介绍，那么我们看看利用DDP来给我们的模型做加速。

在Lightning中使用DDP很简单，只需要在参数做下面的修改

```jsx
TRAINER_ARGS="\
        --max_epochs 5 \
        --gpus 2 \
        --strategy ddp \
        --gradient_clip_val 1.0 \
        --val_check_interval 1.0 \
        --default_root_dir $DEFAULT_ROOT_DIR \
        "
```
用这种方式起来看看。
```
Epoch 0:  10% 446/4564 [02:59<27:36,  2.49it/s, loss=1.17, v_num=107727]
```
可以看到因为用了两张显卡做并行，整体steps由9664减半到4564，训练时间也从60分钟缩短到30分钟，如果大家手头上有多张显卡的话，可以通过数据并行的方式快速做训练加速。


### 单张显卡


大多数开发者手头可能只有单张显卡，并且显存都不大，有没有办法在单卡下进行优化呢？答案是肯定的，比如微软的黑科技[**Deepspeed**](https://github.com/microsoft/DeepSpeed)，通过Zero技术能对训练过程中的中间参数进行优化，节约显存，在单卡上训练、微调大模型成为可能。关于Deepspeed的技术细节在后续的文章中我们也会详细介绍，这里我们先来看看，通过使用Deepspeed能给我们带来怎样的优化吧。
同样是跑到比较稳定的时候看具体需要的时耗，这里使用Deepspeed zero_stage_1，显存优化比较少，但是运行速度是最快的，训练时间缩短了50%，一个epoch只需要30分钟，相对于单卡训练，在batch_size同样为4的情况下，显存使用减少20%
```
Epoch 0:  12% 1136/9664 [03:33<26:40,  5.33it/s, loss=0.947, v_num=107491]
```
![Image](https://pic4.zhimg.com/80/v2-b6248d1fed1c2ff88ed0e9a55f06e09e.png)
上图为内存消耗，单位为MB，其中verson_8是单卡训练，version_107491是使用了Deepspeed后的效果。


Deepspeed不单单能让我们的模型训练更快，更能让我们用更小的显存装下更大的模型，我们试试在Deepspeed的极致优化下，装下我们13亿参数的模型进行训练，需要多大的显存呢？
batchsize改为1，看看显存消耗的情况
```
Epoch 0:   3% 1000/38924 [45:51<28:59:25,  2.75s/it, loss=2.07, v_num=107671]
```
优化开高后明显看到训练时间变长了很多，需要一天多才能完成训练，但是内存使用量大大减少
![Image](https://pic4.zhimg.com/80/v2-d749708e2f2697268ad891cf284498b3.png)
**仅需要5G显存**即可开始Finetune！


那么回到刚刚Deepspeed zero_stage_0，仅需要30分钟即可跑完一个epoch了，在这个基础上我们还有优化空间吗？答案是肯定的，我们可以通过使用更优化的Optimizer来提升我们的训练速度。比如说：

- 谷歌提出的[Lamb](https://arxiv.org/abs/1904.00962)
- 微软提出的[OneBitLamb](https://arxiv.org/abs/2104.06069)
- 微软提出的[OneBitAdam](https://arxiv.org/abs/2102.02888)
- 微软提出的[ZeroOneAdam](https://arxiv.org/abs/2202.06009)

在以后的系列文章中，我们也会为大家一一介绍这些Optimizer的实现以及实际使用的效果，欢迎Follow我们的专栏。


### 总结

通过简单的介绍了一下常用的模型加速方法，大家或许对怎么做训练加速有了一定的想法，但本文对其中的实现细节以及什么情况下应该选取什么样的优化策略还没有进行更深层次的剖析。
通过使用并行训练、Zero等技术，我们能够在低显存、显卡数少的情况下提升我们的训练效率，本文也以13亿参数模型为例，成功将训练时间缩短50%。
本文仅仅是开了一个头，大模型的预训练、微调是个大坑，未来还有很多很多技术深入研究，欢迎同学follow我们的专栏，下一篇将带大家进入挖掘模型并行的实现细节。