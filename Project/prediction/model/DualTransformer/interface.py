import os, sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print("base_path:", base_path)
sys.path.append(base_path)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DualTransformer'))
import json
from tqdm import tqdm
import torch
import torch.optim as optim
import pickle
import time
import warnings
import itertools
import numpy as np
import logging
import copy
from datetime import datetime
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .dataloader import DualTransformerDataLoader
from prediction.model.base.interface import Interface
from model import Model, SoccerDataset
from main import my_load_model, compute_RMSE, display_result
from prediction.dataset.generate import input_data_by_attack_step
logger = logging.getLogger(__name__)


class DualTransformerInterface(Interface):
    def __init__(self, obs_length, pred_length, pre_load_model=None, rescale=[105,68]):
        super().__init__(obs_length, pred_length)
  
        self.dataloader = DualTransformerDataLoader(
            self.obs_length, self.pred_length
        )
        
        self.rescale = rescale
        self.dev = 'cuda:0'
        self.save_path = os.path.join(base_path, "data", "dualtransformer_dfl", "model", "lstm_best.pt")
        self.rnn_type = "lstm" if "lstm" in self.save_path else "gru"
        if pre_load_model is not None:
            self.model = self.load_model(self.default_model(), pre_load_model)
        else:
            if os.path.isfile(self.save_path):
                self.model = self.default_model()
                self.model = self.load_model(self.model, self.save_path)
                print(f"Loaded pretrained weights from {self.save_path}")
            else:
                self.model = self.default_model()
                print("No pretrained model found – start training")
                self.train()   

    def default_model(self):
        model = Model(rnn_type=self.rnn_type)

        model.to(self.dev)
        return model

    def load_model(self, model, model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['seq2seq_model'])
        logger.warn('Successfull loaded from {}'.format(model_path))
        return model

    def save_model(self, model, model_path):
        print(model_path)
        torch.save({'seq2seq_model': model.state_dict()}, model_path)
        logger.warn("Model saved to {}".format(model_path))

    def run(self, input_data, perturbation=None, backward=False):
        if self.model is None:
            self.model.train()

        if not backward:
            self.model.eval()
        
        _input_data, output_loc_GT, rescale_xy, obj_index = self.dataloader.preprocess(input_data, rescale_x=self.rescale[0], rescale_y=self.rescale[1])
        _input_data = _input_data.unsqueeze(0)

        predicted = self.model(_input_data) # (N, C, T, V)=(N, 2, 6, 120)
        output_data, loss = self.dataloader.postprocess(input_data, perturbation, predicted, rescale_xy, obj_index)

        if loss is None:
            return output_data
        else:
            return output_data, loss
        
    def train(self):
        train_path = "./data/dataset/dfl/multi_frame/train"

        inputs = []
        targets = []
        dataloader = DualTransformerDataLoader(
            self.obs_length, self.pred_length
        )

        files = os.listdir(train_path)
        for file in tqdm(files[:], desc="generate training dataset"):
            with open(os.path.join(train_path, file), 'rb') as f:
                input_data = json.load(f) 

            # preprocess 호출
            input_data = input_data_by_attack_step(input_data, self.obs_length, self.pred_length, 0)
            input_tensor, output_loc_GT, rescale_xy, obj_index = dataloader.preprocess(input_data, rescale_x=self.rescale[0], rescale_y=self.rescale[1])
            
            inputs.append(input_tensor.cpu().numpy())   # (T, N, F)
            targets.append(output_loc_GT.cpu().numpy()) # (T, N, 2)

        inputs = np.stack(inputs)   # (total, T, N, F)
        targets = np.stack(targets) # (total, T, N, 2)

        ratio = 0.8
        split = int(len(inputs) * ratio)
        train_inputs,  valid_inputs  = inputs[:split],  inputs[split:]
        train_targets, valid_targets = targets[:split], targets[split:]

        train_set  = SoccerDataset(train_inputs,  train_targets)
        valid_set  = SoccerDataset(valid_inputs,  valid_targets)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)   #

        # -------------------- 3) Loss / Optimizer ----------------------
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        best_val = float("inf")
        for epoch in range(1, 30+1):
            # ---------------- train ----------------
            self.model.train()
            train_loss = 0.0
            for xb, yb in tqdm(train_loader, desc=f"[{epoch}] train", leave=False):
                xb, yb = xb.to(self.dev), yb.to(self.dev) # (B, T, N, F), (B, T, N, 2)
            
                pred = self.model(xb)
                loss = criterion(pred, yb)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss += loss.item()
            train_loss /= len(train_set)

            # ---------------- valid ---------------- ★
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    xb, yb = xb.to(self.dev), yb.to(self.dev)

                    pred = self.model(xb)
                    loss = criterion(pred, yb)

                    val_loss += loss.item() 
            val_loss /= len(valid_set)

            print(f"[epoch {epoch}] train MSE: {train_loss:.5f}  |  valid MSE: {val_loss:.5f}")
            
            # ----------- save best on valid -------- ★
            if val_loss < best_val:
                best_val = val_loss
                self.save_model(self.model, self.save_path)
                print(f"  ↳ new best! model saved to {self.save_path}")

        self.model = self.default_model()
        self.model = self.load_model(self.model, self.save_path)
        valid_loss = []
        for xb, yb in valid_loader:
            xb, yb = xb.to(self.dev), yb.to(self.dev)
            pred = self.model(xb)
   
            # pred *= torch.tensor(self.rescale, dtype=torch.float32).to(self.dev)
            # yb *= torch.tensor(self.rescale, dtype=torch.float32).to(self.dev)

            loss = criterion(pred, yb)
            valid_loss.append(loss.item())

        print(f"Final valid loss: {np.mean(valid_loss):.5f}")
            
