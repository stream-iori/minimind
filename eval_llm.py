import argparse
import random
import os
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed

warnings.filterwarnings('ignore')

def init_model(args):
    """
    初始化模型和分词器
    根据参数选择加载原生MiniMind模型（Pytorch原生权重）或者Transformers格式模型
    """
    # 加载分词器，trust_remote_code=True允许加载远程代码（对于自定义模型是必须的）
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)

    # 如果load_from路径包含'model'，则认为是加载原生Pytorch模型权重（MiniMind原生方式）
    if 'model' in args.load_from:
        # 初始化MiniMind模型配置
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,                   # 隐藏层维度
            num_hidden_layers=args.num_hidden_layers,       # 隐藏层层数
            use_moe=bool(args.use_moe),                     # 是否使用混合专家(MoE)架构
            # TODO RoPE概念需要理解
            inference_rope_scaling=args.inference_rope_scaling # 是否开启RoPE位置编码外推（用于长文本）
        ))

        # 构建模型权重文件路径
        moe_suffix = '_moe' if args.use_moe else ''

        # ckp checkPoint path
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'

        # 加载模型权重
        # torch.load 加载ckp文件
        # 把刚才读到内存里的数据（字典），填入到你现在的 model 代码结构中
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        # 如果指定了LoRA权重，则应用LoRA
        if args.lora_weight != 'None':
            apply_lora(model) # 将LoRA层注入到模型中
            load_lora(model, os.path.join(args.save_dir, 'lora', f'{args.lora_weight}_{args.hidden_size}.pth'))
    else:
        # 否则使用Transformers库加载标准模型格式（如HuggingFace下载的模型）
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)

    # 打印模型参数量（单位：百万/Million）
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')

    # 将模型设置为评估模式，并移动到指定设备（GPU/CPU/MPS）
    return model.eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    # 模型加载相关参数
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录，默认在out目录下寻找")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain=预训练, full_sft=指令微调, rlhf=偏好优化, reason=推理模型, ppo_actor=PPO, grpo=GRPO, spo=SPO）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选如：lora_identity=自我认知, lora_medical=医疗增强）")

    # 模型结构参数（需与训练时保持一致）
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度（512对应Small-26M, 640对应MoE-145M, 768对应Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量（Small/MoE=8层, Base=16层）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（用于处理超过训练长度的输入，仅解决位置编码问题）")

    # 生成参数
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成新token长度（注意：实际受限于模型的上下文窗口）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机，越小越确定）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1，控制采样范围）")
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史，N表示携带最近N条）")

    # 设备选择：优先CUDA，其次MPS（Mac），最后CPU
    default_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    parser.add_argument('--device', default=default_device, type=str, help="运行设备")
    args = parser.parse_args()

    # 预设的测试问题列表
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]

    conversation = [] # 初始化对话历史
    model, tokenizer = init_model(args) # 初始化模型

    # 选择交互模式
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))

    # TextStreamer用于流式输出生成的文本（打字机效果）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 根据模式选择输入源：自动测试列表 或 用户手动输入
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')

    for prompt in prompt_iter:
        # 设置随机种子以保证结果可复现（也可设为随机以增加多样性）
        setup_seed(2026) # or setup_seed(random.randint(0, 2048))

        if input_mode == 0: print(f'👶: {prompt}')

        # 处理对话历史，只保留最近的 args.historys 条消息
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # 准备聊天模板参数
        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason':
            templates["enable_thinking"] = True # 仅Reason推理模型需要开启思考标签

        # 应用聊天模板或仅添加BOS token（预训练模型没有微调过对话模板）
        if args.weight != 'pretrain':
            inputs = tokenizer.apply_chat_template(**templates)
        else:
            inputs = tokenizer.bos_token + prompt

        # 将输入文本转为tensor并移动到指定设备
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖️: ', end='')
        # 开始生成
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, # 最大生成长度
            do_sample=True,                     # 启用采样
            streamer=streamer,                  # 流式输出
            pad_token_id=tokenizer.pad_token_id,# 填充token ID
            eos_token_id=tokenizer.eos_token_id, # 结束token ID
            top_p=args.top_p,                   # 核心采样概率
            temperature=args.temperature,       # 温度
            repetition_penalty=1.0              # 重复惩罚（1.0表示不惩罚）
        )

        # 解码生成的ID为文本，跳过输入部分，仅保留新生成的内容
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

        # 将回答加入历史记录
        conversation.append({"role": "assistant", "content": response})
        print('\n\n')

if __name__ == "__main__":
    main()
