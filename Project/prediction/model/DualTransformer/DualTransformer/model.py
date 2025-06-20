import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, EncoderRNN
import numpy as np 

class SoccerDataset(Dataset):
	def __init__(self, X, Y):
		self.x, self.y = X, Y
	def __len__(self): 
		return len(self.x)
	def __getitem__(self, i): 
		return self.x[i], self.y[i]

class Model(nn.Module):
    def __init__(
        self,
        rnn_type: str = "lstm",
        in_dim: int = 2,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 2,
        output_length: int = 5,
    ):
        super().__init__()
        assert rnn_type in ("lstm", "gru")

        self.rnn_type = rnn_type
        self.output_length = output_length
        self.output_dim = 2
        self.input_fc = nn.Linear(in_dim, embed_dim)
        
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.encoder = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.decoder = rnn_cls(
            input_size=embed_dim,  # 좌표 (x, y) 를 입력으로 받음
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True
        )

        self.out_fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):  # x : (B, T, N, F)
        B, T, N, F = x.shape
        x = (
            x.permute(0, 2, 1, 3) # (B, N, T, F)
            .contiguous()           # (B, N, T, F)
            .view(B * N, T, F)      # (B* N, T, F)
        )
        x = self.input_fc(x)                # (B*N, T, embed_dim)

        if self.rnn_type == "lstm":
            encoder_outputs, (h, c) = self.encoder(x)
        else:  
            encoder_outputs, h = self.encoder(x)
            c = None

        decoder_input = x[:, -1, :]  # 마지막 timestep의 출력만 사용하여 디코더 입력으로 사용
        decoder_input = decoder_input.unsqueeze(1)

        # Decoder: autoregressive
        outputs = []
        for t in range(self.output_length):
            if self.rnn_type == "lstm":
                dec_out, (h, c) = self.decoder(decoder_input, (h, c))
            else:
                dec_out, h = self.decoder(decoder_input, h)
            pred = self.out_fc(dec_out.squeeze(1))  # (B*N, 6)
          
            outputs.append(pred.view(B, N, self.output_dim).contiguous())  # (B, N, 6)
            decoder_input = pred.unsqueeze(1)  # (B*N, 1, 2)
            decoder_input = self.input_fc(decoder_input)  # (B*N, 1, embed_dim)

        outputs = torch.cat(outputs)  # (B*T_pred, N, 6)
        outputs = outputs.view(B, self.output_length, N, self.output_dim)

        return outputs