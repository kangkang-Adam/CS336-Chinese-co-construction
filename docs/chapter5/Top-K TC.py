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
        # 线性投影后得到专家与token匹配得分张量[batch_size, num_experts]
        gate_logits = self.w_g(x)                   
        # 输出每行和为1，表示token对各专家的匹配概率，每个概率都大于0
        gate_scores = F.softmax(gate_logits, dim=-1)   
        # Top-K稀疏化处理，选出每个token概率最高的k个专家
        topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)    
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
        # 初始化路由器，用于计算每个token对每个专家的匹配分数并选择Top-k专家
        self.router = TopKRouter(dim, num_experts, k)
        # 初始化专家集合，每个专家都是一个小前馈网络，输入输出维度为dim
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])  
        self.k = k
        self.num_experts = num_experts
    def forward(self, x, verbose=False, tokens=None):
        # x每一行为一个token的特征向量，形状为[batch_size, dim]
        gate_scores, topk_idx, topk_scores = self.router(x) 
        if verbose:
            # 可视化路由器决策，每个token被分配的专家和对应概率
            self.visualize(tokens, gate_scores, topk_idx, topk_scores)  
        # 初始化输出张量[batch_size, dim]，用于累加所有激活专家的加权输出
        out = torch.zeros_like(x)  
        for i in range(self.k):
            # 当前第i个Top-k专家索引，每个token对应的专家编号
            idx = topk_idx[:, i]
            # 初始化当前Top-k专家输出，用于累加同一个batch内该专家处理的所有token
            expert_output = torch.zeros_like(x)  

            for e_id, expert in enumerate(self.experts):  # 遍历每个专家
                mask = (idx == e_id).float().unsqueeze(1)  # 生成掩码[B, 1]，mask[i]=1表示第i个token属于当前专家
                if mask.sum() > 0:
                    # 当前专家仅处理属于自己的token，其余token被置为0
                    expert_output += expert(x * mask)
            # 对当前Top-k专家的输出按路由概率加权，并累加到最终输出，shape: [B, dim]
            out += expert_output * topk_scores[:, i].unsqueeze(1)  

        # out返回batch内所有token经过MoE层处理后的特征，shape：[batch_size, dim]
        # out是Top-k专家按匹配权重加权后的加工结果，每个维度表示token在处理专家特征空间上的激活度，可理解为“专家视角的语义量化表示”。
        return out 
     # 可视化路由
    def visualize(self, tokens, gate_scores, topk_idx, topk_scores):
        print("\n========== Token → Expert 路由可视化 ==========\n")
        # 将张量从GPU拷贝到CPU，并去掉梯度信息，方便打印
        # 模型权重的计算图，默认自带梯度
        gate_scores = gate_scores.detach().cpu()
        topk_idx = topk_idx.detach().cpu()
        topk_scores = topk_scores.detach().cpu()
        B, E = gate_scores.shape  # B=token数量(batch size)，E=专家数
        for i in range(B):  # 遍历每一个token
            token = tokens[i]  # 当前token的字符表示

            # 打印token及其字节值
            print(f"Token {i}: '{token}' (byte={ord(token) if len(token) == 1 else token})\n")

            # 打印该token对所有专家的softmax匹配分数
            print("  全部专家得分：")
            for e in range(E):
                print(f"    Expert {e:2d} : {gate_scores[i, e]:.4f}")

            # 打印top-k专家与其对应得分
            print("  Top-k 专家：")
            for k in range(topk_idx.size(1)):
                print(f"    Expert {topk_idx[i, k].item()}  score={topk_scores[i, k]:.4f}")

            # 打印专家得分的可视化柱状图
            print("\n  路由图：")
            max_score = gate_scores[i].max().item()  # 归一化到最高得分，用于绘柱状图
            for e in range(E):
                # 柱状图长度按score/max_score比例缩放到最多20格
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
