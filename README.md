一个输入为上联，输出下联的CharRNN。

#### 模型说明

首先使用word-embedding对汉字重新编码到500维向量，之后经过encoderRNN和decoderRNN，其中decoderRNN通过Attention对encoder的最后一个隐藏层输出加权，decoderRNN的第一轮输入为句子起始符SOS。

- 模型使用GRU作为RNNCell，加入了Luong Attention，word-embedding是从随模型共同训练的。
- 由于输出长度不确定，因此引入句子终结符EOS，当decoderRNN输出EOS后就视作完成一次输出。
- 由于RNN很容易出现梯度爆炸，所以使用clipping和GRU作为Cell，不使用LSTM是为了减少参数，加速训练。

#### 文件说明

- 采用科赛上的中国对联训练集，包含77w+的对联，9000+的汉字，保险起见就不发到网上了

- test.py用来测试，输入格式为`python test.py XXXX`，其中XXXX为上联，test.py会输出下联
- RNN.py保存encoder，decoder和Attention的网络结构和训练代码

#### 训练结果

RNN由于具有时序性，所以无法在GPU上很好的加速，因此迭代次数有限，Model文件夹为迭代29epoch后的模型。

以下对联为CharRNN的输出结果（由于每轮起始是GRU中的Memory为随机的，输出也具有随机性）：

##### 1

>上联：<s>天<\s>
下联：<s>地<\s>
上联：<s>雨<\s>
下联：<s>烟<\s>

##### 2

>上联：<s>米饭<\s>
下联：<s>油茶<\s>
上联：<s>山花<\s>
下联：<s>野禽<\s>

##### 3

>上联：<s>鸡冠花<\s>
下联：<s>龙牙梨<\s>
上联：<s>孔夫子<\s>
下联：<s>毛小公<\s>

##### more

>上联：<s>今天打雷下雨<\s>
下联：<s>昨日打人走人<\s>
上联：<s>狗和猫打架不分胜负<\s>
下联：<s>狼与狗进球就是高多<\s>

文字越多输出的连贯性越差，并且可能出现如下字数不相符的情况：

>上联：<s>人生没有彩排，每一天都是现场直播<\s>
下联：<s>世海无多解势，众今岂来地网先争<\s>

个人理解是如果训练次数足够多可以获得更好的结果。