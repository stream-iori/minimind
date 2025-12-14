# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind模型的配置类，继承自Hugging Face的PretrainedConfig。
    这使得模型可以与Hugging Face生态系统（如from_pretrained, push_to_hub等）兼容。
    """
    model_type = "minimind"  # 模型类型标识

    def __init__(
        self,
        # --- 基本模型参数 ---
        dropout: float = 0.0,  # Dropout比例
        bos_token_id: int = 1,  # 开始符ID
        eos_token_id: int = 2,  # 结束符ID
        hidden_act: str = 'silu',  # 激活函数，'silu'即Swish
        hidden_size: int = 512,  # 隐藏层维度
        intermediate_size: int = None,  # FFN（前馈网络）的中间层维度，若为None则自动计算
        max_position_embeddings: int = 32768,  # 模型支持的最大序列长度
        num_attention_heads: int = 8,  # 注意力头的数量
        num_hidden_layers: int = 8,  # Transformer层数
        num_key_value_heads: int = 2,  # K和V头的数量（用于Grouped-Query Attention, GQA）
        vocab_size: int = 6400,  # 词汇表大小
        rms_norm_eps: float = 1e-05,  # RMSNorm中的epsilon，防止除以零
        rope_theta: int = 1000000.0,  # RoPE（旋转位置编码）的基数
        inference_rope_scaling: bool = False,  # 是否在推理时启用RoPE外推（用于处理超长序列）
        flash_attn: bool = True,  # 是否使用Flash Attention（需要PyTorch 2.0+）

        # --- MoE (Mixture of Experts) 特定配置 ---
        # 当 use_moe 为 False 时，以下参数无效
        use_moe: bool = False,  # 是否启用MoE架构
        num_experts_per_tok: int = 2,  # 每个token（词元）路由到的专家数量
        n_routed_experts: int = 4,  # 总的可路由专家数量
        n_shared_experts: int = 1,  # 共享专家的数量（暂未在模型中完全实现）
        scoring_func: str = 'softmax',  # 门控网络中计算专家分数的函数
        aux_loss_alpha: float = 0.01,  # MoE辅助损失（负载均衡损失）的权重
        seq_aux: bool = True,  # 是否在序列级别计算辅助损失
        norm_topk_prob: bool = True,  # 是否对top-k专家的概率进行归一化
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling

        # 如果启用RoPE外推，则配置YaRN（Yet another RoPE extensioN method）缩放参数
        # 这允许模型处理比训练时更长的序列
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,  # 缩放因子，可以将上下文长度扩展16倍
            "original_max_position_embeddings": 2048,  # 原始训练时的最大长度
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        self.flash_attn = flash_attn

        # --- MoE 配置 ---
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) 的实现。
    比LayerNorm更简单、计算更快。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        # 核心计算：x / sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 将输入归一化后，乘以可学习的权重
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int, rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算RoPE（旋转位置编码）所需的cos和sin值。
    RoPE通过将位置信息编码到复数空间，以旋转的方式注入到Query和Key中，从而实现相对位置编码。
    """
    # 计算基础频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    # 如果启用了YaRN外推，则对频率进行调整
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        if end / orig_max > 1.0:  # 只有当推理长度超过原始训练长度时才激活
            beta_fast = rope_scaling.get("beta_fast", 32.0)
            beta_slow = rope_scaling.get("beta_slow", 1.0)
            attn_factor = rope_scaling.get("attention_factor", 1.0)

            # YaRN的核心思想：对高频和低频部分应用不同的缩放策略，以减少信息损失
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0,
                               1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 计算不同位置（t）和不同频率（freqs）的相位角
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 计算cos和sin值，并复制以匹配维度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将预计算好的RoPE（cos和sin）应用到Query和Key上。
    """

    def rotate_half(x):
        # 将特征维度分成两半，并交换它们的位置，实现复数乘法中的旋转
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # q_embed = q * cos + rotate_half(q) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    为Grouped-Query Attention (GQA) 重复Key和Value。
    在GQA中，多个Query头共享同一组Key和Value头，此函数将KV头重复n_rep次以匹配Q头的数量。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen,
                                                                                           num_key_value_heads * n_rep,
                                                                                           head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力机制（支持GQA和Flash Attention）。
    """

    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # GQA中每个KV头被共享的次数
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 线性投射层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 检查是否支持并启用Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 接收预计算的cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape

        # 1. 线性投射
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 2. 应用RoPE旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 3. KV Cache机制（用于高效生成）
        if past_key_value is not None:
            # 将过去的K, V与当前的K, V拼接起来
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 4. GQA: 重复KV头以匹配Q头
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 5. 计算注意力分数
        # 如果支持Flash Attention且满足条件，则使用优化后的内核
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=True)
        else:
            # 标准的点积注意力计算
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 应用因果掩码（Causal Mask），防止未来的token影响当前token
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                # 应用padding mask
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        # 6. 整合输出
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    标准的前馈网络（FFN），采用SwiGLU激活函数。
    FFN(x) = Dropout(W_down * (Swish(W_gate * x) * (W_up * x)))
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 自动计算中间层维度，通常是隐藏层维度的倍数
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)  # 确保是64的倍数
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # SwiGLU: Swish(gate_proj(x)) * up_proj(x)
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoE的门控网络（Gating Network）。
    负责为每个输入token计算一组权重，决定将该token路由到哪些专家网络。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择k个专家
        self.n_routed_experts = config.n_routed_experts  # 总专家数

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha  # 辅助损失权重
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        # 门控网络的权重，将hidden_states映射到每个专家的得分
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)

        # 1. 计算每个token到每个专家的逻辑值（logits）
        logits = F.linear(hidden_states, self.weight, None)

        # 2. 将logits转换为概率分数
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 3. 选出得分最高的 top-k 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 4. （可选）对 top-k 专家的权重进行归一化
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # 5. 计算辅助损失（Load Balancing Loss）
        # 这个损失鼓励门控网络将负载均匀地分配给所有专家，防止部分专家过载而其他专家空闲
        aux_loss = 0
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                # 在序列维度上计算辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # 在token维度上计算辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    混合专家（MoE）前馈网络。
    它包含多个独立的FFN（专家），并通过一个门控网络来动态地为每个token选择一部分专家进行计算。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建多个专家网络
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)  # 门控网络
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape

        # 1. 通过门控网络获取专家的索引和权重
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        # 训练和推理使用不同的路径以优化性能
        if self.training:
            # 训练时，为了并行计算，将输入复制k次，每个副本由一个被选中的专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                # 找到所有应该由当前专家处理的token
                mask = (flat_topk_idx == i)
                if mask.any():
                    # 让专家处理对应的token
                    y[mask] = expert(x[mask]).to(y.dtype)

            # 将k个专家的输出按权重加权求和
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时，为了节省显存，采用一种更高效的逐个专家计算方式
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts > 0:
            # (暂未实现) 添加共享专家的输出
            for expert in self.shared_experts:
                y = y + expert(identity)

        self.aux_loss = aux_loss  # 保存辅助损失，以便在最终的loss中计入
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        """
        MoE的高效推理实现。
        它将token按其分配的专家进行分组，然后批量送入每个专家，最后将结果聚合。
        这避免了在训练时使用的大量重复张量，从而节省了显存。
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()  # 按专家索引排序
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok

        # 逐个专家进行计算
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue  # 该专家没有被分配到任何token

            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]  # 获取分配给该专家的token的原始索引
            expert_tokens = x[exp_token_idx]  # 提取这些token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 计算专家输出
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 乘以专家权重

            # 将加权后的输出加回到它们在原始序列中的位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    一个完整的Transformer块（或层）。
    包含一个自注意力模块和一个前馈网络（FFN或MoE）。
    采用"Pre-Norm"结构：Norm -> Attention -> Residual -> Norm -> FFN -> Residual
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states

        # 1. 注意力部分（Pre-Norm + Attention + Residual）
        attn_input = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            attn_input, position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # 2. FFN/MoE部分（Pre-Norm + MLP + Residual）
        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(mlp_input)

        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind模型的主体，由多个Transformer块堆叠而成。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers

        # 词嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 训练时丢弃一些，让他学的更全面一点，避免过拟合
        self.dropout = nn.Dropout(config.dropout)

        # 堆叠的Transformer层, 也就是Transformer Block
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])

        # 最终的归一化层，Block外面的最后一个Norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算RoPE的cos和sin值，并注册为buffer（不作为模型参数）
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                # 这里的输入。比如 [101, 204, 305]，代表“我喜欢猫”
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape

        # 处理KV Cache
        # 如果是第一次读（Training 或生成第1个词）：日记是空的，start_pos = 0。我们要从头开始读
        # Inference 日记里已经记了“我喜欢”，现在要生成“猫”。start_pos 就是 3
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 1. 获取词嵌入
        # 上面init的时候已经给了embed_tokens这个方法，这里输入的是token ids, 得到的是一个嵌入向量
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 2. 获取当前序列所需的位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 3. 逐层通过Transformer Block
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 4. 最终归一化
        hidden_states = self.norm(hidden_states)

        # 5. 收集所有MoE层的辅助损失
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward) and hasattr(layer.mlp, 'aux_loss')
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    最终的因果语言模型（Causal LM）。
    在MiniMindModel的基础上增加了一个语言模型头（LM Head），用于预测下一个token的概率。
    继承了Hugging Face的PreTrainedModel和GenerationMixin，从而获得了完整的生成能力（如.generate()方法）。
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 语言模型头，将隐藏状态映射到词汇表大小
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重绑定：将词嵌入层和LM头的权重绑定在一起。这是一种常见的做法，可以减少参数量并提高性能。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                labels: Optional[torch.Tensor] = None,  # `labels`参数用于计算损失
                **args):
        # 调用主模型获取最终的隐藏状态
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # 通过LM头计算logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 如果提供了labels，则计算交叉熵损失
            # 将logits和labels移动到同一设备
            logits = logits.to(labels.device)
            # 将logits和labels的形状调整为 (batch_size * seq_len, vocab_size) 和 (batch_size * seq_len)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 使用交叉熵损失函数
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # 计算主损失
            loss = loss_fct(shift_logits, shift_labels)
            # 如果有MoE辅助损失，则加到总损失中
            if aux_loss is not None:
                loss += aux_loss

        # 返回符合Hugging Face格式的输出对象
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )
