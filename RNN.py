import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv
import random


from google.colab import drive
drive.mount('/content/gdrive')


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
	print('GPU Avaliable')
device = torch.device("cuda" if USE_CUDA else "cpu")
#learning_rate = 0.001

epoch_num = 100

SOS = 1
EOS = 0
attn_model = 'dot'
encoder_n_layers = 5
decoder_n_layers = 5
dropout = 0.1
batch_size =128
#数据集中一联的最大长度
maxlen = 34
#word-embedding后字符向量的长度
hidden_size = 500

load_from_file = True
start_epoch = 23

#用0代表句子结束，获取所有的汉字集合
def getMap():
	chmap = {}
	reversemap = {}
	count = 1
	text = open('./vocabs.txt','r',encoding='utf-8')
	row = csv.reader(text,delimiter=' ')
	i = 0
	for r in row:
		count += 1
		chmap[r[0]]=count
		reversemap[count]=r[0]
	return chmap,reversemap


def getdata(chmap):
	sample_in=[]
	sample_out=[]
	text_in = open('./train_in.txt','r',encoding='utf-8')
	text_out = open('./train_out.txt','r',encoding='utf-8')
	row_in = csv.reader(text_in, delimiter=' ')
	row_out = csv.reader(text_out, delimiter=' ')
	for r in row_in:
		sentence = [SOS]
		for ch in r:
			if ch:
				sentence.append(chmap[ch])
		#add EOS
		sentence.append(EOS)
		sample_in.append(sentence)
	for r in row_out:
		sentence = []
		for ch in r:
			if ch:
				sentence.append(chmap[ch])
		#add EOS
		sentence.append(EOS)
		sample_out.append(sentence)

	return sample_in,sample_out


#RNN的输入格式是(序列长度，batch_size，数据)
def Sample2Batch(sample):
	sample.sort(key=lambda x:len(x), reverse=True)
	batch_num = int(len(sample)/batch_size)
	batchs=[]
	mask=[]
	lengthlist = []
	for bdx in range(batch_num):
		new_batch = np.zeros((maxlen,batch_size))
		new_mask = np.zeros((maxlen,batch_size))
		blength = []
		for sbdx in range(batch_size):
			blength.append(len(sample[bdx*batch_size+sbdx]))
			for idx in range(maxlen):
				if idx < len(sample[bdx*batch_size+sbdx]):
					new_batch[idx][sbdx] = sample[bdx*batch_size+sbdx][idx]
					new_mask[idx][sbdx] = 1
		lengthlist.append(blength)
		batchs.append(new_batch)
		mask.append(new_mask)
	return batchs,lengthlist,mask,batch_num



#双向GRU的编码器，输出为最后一个隐藏层的数据
class EncoderRNN(nn.Module):
	def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
		super(EncoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
		# Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
		#   because our input size is a word embedding with number of features == hidden_size
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
						  dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

	def forward(self, input_seq, input_lengths, hidden=None):
		# use word-embedding to preprocess input charactors
		embedded = self.embedding(input_seq)
		# 转化为变长的padding
		packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		outputs, hidden = self.gru(packed, hidden)
		# Unpack padding
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
		# 双向RNN输出直接做和作为输出
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
		return outputs, hidden



# Luong attention layer
class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(hidden_size))

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim=2)

	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden * energy, dim=2)

	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)

	def forward(self, hidden, encoder_outputs):
		# Calculate the attention weights (energies) based on the given method
		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)

		# Transpose max_length and batch_size dimensions
		attn_energies = attn_energies.t()

		# Return the softmax normalized probability scores (with added dimension)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)


#使用Luong Attention的Decoder
class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()

		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout

		# Define layers
		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)

		self.attn = Attn(attn_model, hidden_size)

	def forward(self, input_step, last_hidden, encoder_outputs):
		# Note: we run this one step (word) at a time
		# embedding SOS
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)
		# Forward through unidirectional GRU
		rnn_output, hidden = self.gru(embedded, last_hidden)
		# 计算Attention Weight
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# 计算encoder output基于Attention Weight的加权和
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# 合并encoder output和GRU第一轮的输出
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		# 将word embedding 转化回字符
		output = self.out(concat_output)
		output = F.softmax(output, dim=1)
		# Return output and final hidden state
		return output, hidden




#不去关注padding部分的loss，因为输入的padding部分都是0
def maskNLLLoss(inp, target, mask):
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
	loss = crossEntropy.mean().to(device)
	return loss, nTotal.item()



#在一个batch上训练
def OneBatchTrain(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
		  encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=maxlen):
	#init
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	input_variable = input_variable.to(device)
	lengths = lengths.to(device)
	target_variable = target_variable.to(device)
	mask = mask.to(device)
	loss = 0
	print_losses = []
	n_totals = 0

	# Forward pass through encoder
	encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
	# Create initial decoder input (start with SOS tokens for each sentence)
	decoder_input = torch.LongTensor([[SOS for _ in range(batch_size)]])
	decoder_input = decoder_input.to(device)

	# Set initial decoder hidden state to the encoder's final hidden state
	decoder_hidden = encoder_hidden[:decoder.n_layers]

	# teacher_forcing即将target中第t个字符作为decoder的第t+1轮输入（第一轮为SOS），可以加速训练，不过可能使训练不稳定
	# 以 teacher_forcing_ratio的概率使用teacher_forcing方法
	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	# Forward batch of sequences through decoder one time step at a time
	if use_teacher_forcing:
		for t in range(max_target_len):
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden, encoder_outputs
			)
			# Teacher forcing: next input is current target
			decoder_input = target_variable[t].view(1, -1)
			# Calculate and accumulate loss
			mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
			# when all mask are 0, loss is nan
			if not torch.isnan(mask_loss):
				loss += mask_loss
				print_losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal
	else:
		for t in range(max_target_len):
			decoder_output, decoder_hidden = decoder(
				decoder_input, decoder_hidden, encoder_outputs
			)
			# No teacher forcing: next input is decoder's own current output
			_, topi = decoder_output.topk(1)
			decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
			decoder_input = decoder_input.to(device)
			# Calculate and accumulate loss
			mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
			if not torch.isnan(mask_loss):
				loss += mask_loss
				print_losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal

	# Perform backpropatation
	loss.backward()

	# Clip gradients: gradients are modified in place
	_ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
	_ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

	# Adjust model weights
	encoder_optimizer.step()
	decoder_optimizer.step()

	return sum(print_losses) / n_totals



def saveModel(epoch):
	torch.save(encoder.state_dict(),'./gdrive/My Drive/RNN/'+str(epoch)+'encoder.pth')
	torch.save(decoder.state_dict(),'./gdrive/My Drive/RNN/'+str(epoch)+'decoder.pth')
	torch.save(embedding.state_dict(),'./gdrive/My Drive/RNN/'+str(epoch)+'embedding.pth')
	torch.save(encoder_optimizer.state_dict(),'./gdrive/My Drive/RNN/'+str(epoch)+'eopt.pth')
	torch.save(decoder_optimizer.state_dict(),'./gdrive/My Drive/RNN/'+str(epoch)+'dopt.pth')


def viewOut():
	couplet = random.randint(0,10000)
	testin = raw_in[couplet]
	print('\t<s>',end='')
	for x in testin:
		if x!=EOS and x!=SOS:
			print(reversemap[x],end='')
	print('<\\s>')
	inseq = torch.LongTensor([0 for _ in range(maxlen)]).to(device)
	for i in range(len(testin)):
		inseq[i]=testin[i]
	inseq = inseq.view(maxlen,1)
	length = torch.tensor([len(testin)]).to(device)
	encoder_outputs, encoder_hidden = encoder(inseq,length)
	decoder_hidden = encoder_hidden[:decoder.n_layers]
	decoder_input = torch.LongTensor([[SOS]])
	decoder_input = decoder_input.to(device)
	testout = []
	for t in range(maxlen):
		decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
		testout.append(torch.max(decoder_output,dim=1)[1].item())
	print('\t<s>',end='')
	for x in testout:
		if x==SOS:
			print('<s>',end='')
		if x==EOS:
			print('<\\s>',end='')
		if x!=EOS and x!=SOS:
			print(reversemap[x],end='')
	print('<\\s>')
	


def train():
	for epoch in range(epoch_num):
		sequence = np.random.permutation(range(batch_num))
		bcount = 0
		for bdx in sequence:
			bloss = OneBatchTrain(sample_in[bdx],lengthlist[bdx],sample_out[bdx],mask[bdx],
				maxlen,encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip,maxlen)
			bcount += 1
			print(bcount,end=', ')
			if bcount%300==0:
				print('  {}/{}batch, MaskLoss:{:.8f}'.format(bcount ,batch_num ,bloss))
				for _ in range(5):
					viewOut()
				print('*'*50)
		print('epoch:{}/{}'.format(start_epoch+epoch+1,start_epoch+epoch_num+1))
		saveModel(start_epoch+epoch+1)


chmap,reversemap = getMap()

raw_in,raw_out = getdata(chmap)
sample_in,lengthlist,mask,batch_num = Sample2Batch(raw_in)
sample_out,_,_,_ = Sample2Batch(raw_out)

sample_in = torch.LongTensor(sample_in).to(device)
sample_out = torch.LongTensor(sample_out).to(device)
lengthlist = torch.tensor(lengthlist).to(device)
mask = torch.ByteTensor(mask).to(device)



clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0


#embedding的字符数为|{字符集, SOS, EOS}|
embedding = nn.Embedding(len(chmap)+2, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
encoder = encoder.to(device)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, len(chmap)+2, decoder_n_layers, dropout)
decoder = decoder.to(device)
encoder.train()
decoder.train()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

if load_from_file:
	embedding.load_state_dict(torch.load('./gdrive/My Drive/RNN/'+str(start_epoch)+'embedding.pth'))
	encoder.load_state_dict(torch.load('./gdrive/My Drive/RNN/'+str(start_epoch)+'encoder.pth'))
	decoder.load_state_dict(torch.load('./gdrive/My Drive/RNN/'+str(start_epoch)+'decoder.pth'))
	encoder_optimizer.load_state_dict(torch.load('./gdrive/My Drive/RNN/'+str(start_epoch)+'eopt.pth'))
	decoder_optimizer.load_state_dict(torch.load('./gdrive/My Drive/RNN/'+str(start_epoch)+'dopt.pth'))

train()
viewOut()