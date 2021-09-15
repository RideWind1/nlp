from torch import nn
import torch

class MultiheadAttention(nn.Module):
    #n_heads:多头注意力的数量
    #hid_dim:多个词输出的向量维度
    def __init__(self,hid_dim,n_heads,dropout):
        super(MultiheadAttention,self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        #强制hid_dim必须整除n_heads
        assert hid_dim % n_heads == 0
        #定义W_q矩阵
        self.w_q = nn.Linear(hid_dim,hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim,hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim,hid_dim)
        self.fc = nn.Linear(hid_dim,hid_dim)
        self.do = nn.Dropout(dropout)
        #缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self,query,key,value,mask=None):
        #Q,K,V在句子长度这一个维度的数值可以一样，可以不一样
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        #把K,Q,V矩阵拆分为多组注意力
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        # 将上面得到的Q,K,V进行重新拼接
        Q = Q.view(bsz,-1,self.n_heads,self.hid_dim // self.n_heads).permute(0,2,1,3)
        print(Q[0].shape)
        K = K.view(bsz,-1,self.n_heads,self.hid_dim // self.n_heads).permute(0,2,1,3)
        V = V.view(bsz,-1,self.n_heads,self.hid_dim // self.n_heads).permute(0,2,1,3)

        # 第一步，Q乘以K的转置，除以scale,计算attention
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        attention = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale

        # 如果mask不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10，这里用“0”来指示哪些位置的词向量不能被attention到，比如padding位置，当然也可以用“1”或者其他数字来指示，主要设计下面2行代码的改动
        if mask is not None:
            attention = attention.masked_fill(mask == 0,-1e10)

        # 第二步:计算上一步结果的softmax，再经过dropout，得到attention
        # 这里是对最后一维做softmax,也就是在输入序列的维度做softmax
        # attention: [64,6,12,10]

        attention = self.do(torch.softmax(attention,dim=-1))
        
        # 第三步,attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention,V)

        # 因为query有12歌次，所以把12放到前面，把50歌6放到前面，方便下面拼接多组的结果
        # 对于contiguous的解释 https://blog.csdn.net/kdongyi/article/details/108180250
        x = x.permute(0,2,1,3).contiguous()
        # 这里的矩阵转换就是:把多组注意力的结果拼接起来
        # 最终结果就是[64,12,6,50] -> [64,12,300]
        x = x.view(bsz,-1,self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

# batch_size 为64，有12个词，每个词的Query向量是300维
query = torch.rand(64,12,300)
# batch_size 为64，有10个词，每个词的key向量是300维
key = torch.rand(64,10,300)
# batch_size 为64，有10个词，每个词的Value向量是300维
value = torch.rand(64,10,300)
attention = MultiheadAttention(hid_dim=300,n_heads=6,dropout=0.1)
output = attention(query,key,value)
print(output[0].shape)



  












        




        






        

