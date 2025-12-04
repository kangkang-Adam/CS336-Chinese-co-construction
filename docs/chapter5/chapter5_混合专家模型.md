# 第五章 专家混合模型
专家混合模型（MoE）是当前LLM领域中一项至关重要的技术，它有效地解决了模型规模与计算成本之间的矛盾。这种机制允许模型在不显著增加训练和推理计算量的前提下，大幅扩展其总参数规模和表达能力，从而实现了模型容量（参数数量）与计算效率之间的动态平衡。正是凭借MoE这一机制，Switch Transformer、DeepSeek等模型得以问世并展现出卓越性能。然而，MoE在实际应用中会有负载均衡、跨设备并行、训练不稳定和路由机制设计等工程挑战。接下来，我们将剖析MoE的核心概念、工作原理以及实际应用，并提供解决这些工程挑战的实用思路，旨在以更高的计算效率，正确应用这一机制来扩展LLM的能力。

## 5.1 分析MoE
混合专家模型通过将原本的单一前馈网络（如MLP、FFN）替换为由多个并行子网络组成的专家集合，并通过路由机制在每次计算中仅激活少数专家，从而在保持单次前向计算量（FLOPs）基本不变的前提下显著提升模型的参数容量与表达能力。其核心思想是：模型总体包含大规模参数，但每个输入只使用其中一小部分专家，使得**容量大但计算稀疏**。

**概念直观理解：**
`MoE模型`就像一个拥有大量书籍的图书馆（专家集合）,当一个读者（输入数据）来访时，他并不需要翻遍所有书架，而是根据自己感兴趣的主题由“图书管理员”（路由机制）指引，只去相关书架寻找书籍区域（稀疏激活）。这样一来，读者仍然可以访问图书馆中所有书籍的知识（模型的巨大参数容量），同时查找过程却快速高效（FLOPs）。

### 5.1.1 路由机制与负载均衡

路由机制在一次前向传播中，从所有专家中选择出少量最适合当前输入的专家共同参与计算，通常也被称为门控机制，接下来介绍的是基于 $Top-K$ 实现的`词元选择模式（TC）`和`专家选择模式（EC）`的两种路由机制。

假设一共有 $N$ 个专家，输入为 $x$ ，门控函数为 $G(\cdot)$ ，用于决定每个专家的权重， $E_i(\cdot)$ 表示第 $i$ 个专家的输出，则TC和EC的通用门控机制核心计算公式为：

$$
y = \sum_{i \in \mathcal{T}} G_i(x) E_i(x)
$$

**关键点在于集合 $\mathcal{T}$：**
实现稀疏化的关键步骤，这里的 $\mathcal{T}$ 是通过 **$Top\text{-}k$** 机制选出的索引集合。无论是TC还是EC $G(x)$ 都有的计算过程包含两个步骤：

   - **打分：** 计算路由分数 $h(x) = x \cdot W_g$ 。
   - **稀疏化：** 仅保留分数最高的 $k$ 个专家（激活 $k$ 个专家），并对这些分值进行Softmax归一化。未被选中的 $N-k$ 个专家权重被强制置零，这意味着这部分专家在本次前向传播中*完全不参与计算*，从而在模型参数巨大的情况下，保证了实际FLOPs的高效。

> $W_g$ 是路由器中的可学习线性投影层，它将每个token的特征映射为与专家数量相同的得分向量（logits），表示token与各个专家的匹配程度，模型会对这些得分应用 $top-k$ 策略：在TC模式下，每个token选择最适合处理的专家；在EC模式下，每个专家主动选择最适合处理的token。

<div align="center">
<img width="1000" height="520" alt="c5b3cdd83a238e8a6484d51975277f8a" src="https://github.com/user-attachments/assets/648d1892-b01e-4d40-9c2b-50478d2eeccf" />
   <p>图5.1 词元选择模式</p>
 </div>

- `TC`中 $W_g$ ：在打分步骤中，它可以理解为一个 “专家特长档案”。它将token的隐藏特征映射到专家集合的能力空间，并告诉token不同专家分别擅长什么语义，每个token会根据与各个专家“特长档案”的匹配程度，主动挑选最适合处理自己的Top-K专家。

**简易的 $Top-k$ 词元选择模式实现：**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

#  按字节切分输入
def byte_tokenize(text):
    byte_list = list(text.encode("utf-8"))
    return byte_list

#  简单可学习Byte Embedding
class ByteEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.embed = nn.Embedding(256, dim)  # 0~255 字节

    def forward(self, byte_ids):
        return self.embed(byte_ids)

#  MoE路由器（Top-k）
class TopKRouter(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        self.k = k                                  # 选择专家数量
        self.w_g = nn.Linear(dim, num_experts)      # 可学习线性层，将token特征映射到专家数量维度
    def forward(self, x):
        gate_logits = self.w_g(x)                   # 线性投影后得到专家与token匹配得分张量[batch_size, num_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)   # 输出每行和为1，表示token对各专家的匹配概率，每个概率都大于0
        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)    # 稀疏化处理，选出每个token概率最高的k个专家
        # topk_scores:[batch_size, k]，对应Top-k专家的概率
        # topk_idx:[batch_size, k]，对应Top-k专家的索引
        return gate_scores, topk_idx, topk_scores

#  简单Expert
class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),  # 线性层
            nn.ReLU(),                # 非线性激活函数
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        return self.ffn(x)


#  MoE层
class SimpleMoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        self.router = TopKRouter(dim, num_experts, k)   # 初始化路由器，用于计算每个token对每个专家的匹配分数并选择Top-k专家
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])   # 初始化专家集合，每个专家都是一个小前馈网络，输入输出维度为dim
        self.k = k
        self.num_experts = num_experts
    def forward(self, x, verbose=False, tokens=None):
        # x每一行为一个token的特征向量，形状为[batch_size, dim]
        gate_scores, topk_idx, topk_scores = self.router(x) 
        if verbose:
            self.visualize(tokens, gate_scores, topk_idx, topk_scores)  # 可视化路由器决策，每个token被分配的专家和对应概率

        out = torch.zeros_like(x)  # 初始化输出张量[batch_size, dim]，用于累加所有激活专家的加权输出

        for i in range(self.k):
            idx = topk_idx[:, i]   # 当前第i个Top-k专家索引，每个token对应的专家编号
            expert_output = torch.zeros_like(x)  # 初始化当前Top-k专家输出，用于累加同一个batch内该专家处理的所有 token

            for e_id, expert in enumerate(self.experts):  # 遍历每个专家
                mask = (idx == e_id).float().unsqueeze(1)  # 生成掩码[B, 1]，mask[i]=1表示第i个token属于当前专家
                if mask.sum() > 0:
                    expert_output += expert(x * mask)  # 当前专家仅处理属于自己的token，其余token被置为0
            out += expert_output * topk_scores[:, i].unsqueeze(1)  # 对当前Top-k专家的输出按路由概率加权，并累加到最终输出，shape: [B, dim]

        # out返回batch内所有token经过MoE层处理后的特征，shape：[batch_size, dim]
        return out  # out本质为该token经Top-k专家加工后的特征表示，每个维度可以看作token在特定特征空间的激或“语义量化”

    def visualize(self, tokens, gate_scores, topk_idx, topk_scores):
        print("\n========== Token → Expert 路由可视化 ==========\n")
        gate_scores = gate_scores.detach().cpu()
        topk_idx = topk_idx.detach().cpu()
        topk_scores = topk_scores.detach().cpu()
        B, E = gate_scores.shape
        for i in range(B):
            token = tokens[i]
            print(f"Token {i}: '{token}' (byte={ord(token) if len(token)==1 else token})\n")
            print("  全部专家得分：")
            for e in range(E):
                print(f"    Expert {e:2d} : {gate_scores[i, e]:.4f}")
            print("  Top-k 专家：")
            for k in range(topk_idx.size(1)):
                print(f"    Expert {topk_idx[i, k].item()}  score={topk_scores[i, k]:.4f}")
            print("\n  路由图：")
            max_score = gate_scores[i].max().item()
            for e in range(E):
                bar = "█" * int((gate_scores[i, e] / max_score) * 20)
                print(f"    E{e}: {bar}")
            print("\n----------------------------------------------------")

if __name__ == "__main__":
    sentence = "MoE是很强大的机制！"
    print("输入句子：", sentence)
    # 1. 字节切分
    byte_ids = byte_tokenize(sentence)
    tokens = []
    byte_list = sentence.encode("utf-8")
    i = 0
    for ch in sentence:
        utf8_bytes = ch.encode("utf-8")  # 该字符占几个字节
        for _ in utf8_bytes:
            tokens.append(ch)  # 每个字节都对应同一个字符
    byte_ids = torch.tensor(byte_ids)
    print("字节token:", byte_ids.tolist())

    # 2. token编码
    embed = ByteEmbedding(dim=32)
    x = embed(byte_ids)

    # 3. MoE
    moe = SimpleMoE(dim=32, num_experts=6, k=2)
    out = moe(x, verbose=True, tokens=tokens)

    print("MoE输出张量形状:", out.shape)
```
<div align="center">
<img width="1000" height="773" alt="1980610e1c138e069cd6fdcdc3b196a3" src="https://github.com/user-attachments/assets/d665c6bd-88be-4b35-9199-71dbfe74b9ba" />
   <p>图5.2 专家选择模式</p>
 </div>  

- `EC`中 $W_g$ ：在打分步骤中，它可以理解为一个 “语义导航器”，将token的隐藏特征映射到每个专家的语义空间，并把这份导航信号提供给所有专家。每个专家会根据这份“导航信息”，主动挑选最符合自己能力范围的Top-K token进行处理。

因此，在每一次前向传播中，模型只会对 $Top-K$ 的路由机制挑选出的专家子集 $T$ 进行计算，从而实现稀疏化推理，路由机制的核心作用可以概括为**每个输入挑选最合适的少数专家+对激活专家的输出进行加权**。值得注意的是，路由机制这个过程中关于路由依据：这是基于隐藏状态的计算结果。词元经过位置编码等处理后生成隐藏状态，随后输入MLP、FFN等中进行处理；对于专家选择模式，虽然处理的是同一组词元序列，但其排序选择函数运作在行维度。





强制实现了专家间的负载均衡。这种设计在分布式计算场景中优势明显——当专家部署在不同设备时，能确保计算资源均衡利用。关于路由逻辑的直观理解：词元选择模式下，评分函数衡量的是词元与专家的匹配度，自动选择最优专家；而专家选择模式则保障每个专家处理相同数量的词元，两种方案各有优劣，需要根据具体场景权衡。

### 5.1.2 训练稳定性
### 5.1.3 混合专家与稠密模型
## 5.2 MoE的应用

  ### 5.2.1 MoE与Transformer
  ### 5.2.2 MoE与LLM
  ### 5.2.3 推理与部署
## 5.3 DeepSeek创新与实战复现

  ### 5.3.1 DeepSeek的创新关键点
  ### 5.3.2 小规模复现实验
## 5.4 近期MoE研究
和其他领域的联系。
## 思考
## 参考文献

