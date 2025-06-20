import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DualTransformer'))
import logging
import pickle
import random
import torch
from scipy import spatial 
from prediction.model.base.dataloader import DataLoader
from layers.graph import Graph
import numpy as np
from prediction.model.utils import detect_tensor, smooth_tensor


class DualTransformerDataLoader(DataLoader):
    def __init__(self, obs_length=10, pred_length=5):
        super().__init__(obs_length, pred_length)
        self.max_agents = 23
        self.fs = 6
        self.ls = 2
        self.dev = 'cuda:0' 

    def preprocess(self, input_data, rescale_x=105, rescale_y=68):
        # rescale_x = 1
        # rescale_y = 1
        rescale_xy = torch.ones((1,self.pred_length,self.max_agents,self.ls)).to(self.dev)
        rescale_xy[:, :, :, 0] = float(rescale_x)
        rescale_xy[:, :, :, 1] = float(rescale_y)

        visible_object_id_list = []
        for obj_id, obj in input_data["objects"].items():
            if obj["visible"]:
                visible_object_id_list.append(obj_id)
       
        obj_index = {_obj_id:index for index, _obj_id in enumerate(visible_object_id_list)}
        
        # DualTransformer only maintains visible objects
        object_feature_list = []
        object_label_list = []
        for obj_id, obj in input_data["objects"].items():
            if obj["visible"]:
                feature = input_data["objects"][obj_id]["observe_feature"]  # (obs_len, 6)
                #label = input_data["objects"][obj_id]["future_trace"]
                label = input_data["objects"][obj_id]["future_feature"]  # (pred_len, 6)

                if np.isnan(label).any():
                    continue
                
                feature = np.nan_to_num(feature, nan=0.0)  # NaN 있으면 0으로 대체
                object_feature_list.append(feature)  # (obs_len, 6)
                object_label_list.append(label)

        object_feature_array = np.stack(object_feature_list, axis=0) # (num_visible_object, obs_len, 6)
        object_label_array = np.stack(object_label_list, axis=0)     # (num_visible_object, pred_len, 2)

        # # 만약 전체 에이전트 수 (self.max_num_object=23)보다 적으면 패딩
        # if object_feature_array.shape[0] < self.max_agents:
        #     pad_shape = (self.max_agents - object_feature_array.shape[0], self.obs_length, self.fs)
        #     padding = np.zeros(pad_shape, dtype=np.float32)
        #     object_feature_array = np.concatenate([object_feature_array, padding], axis=0)

        # # 만약 전체 에이전트 수 (self.max_num_object=23)보다 적으면 패딩
        # if object_label_array.shape[0] < self.max_agents:
        #     pad_shape = (self.max_agents - object_label_array.shape[0], self.pred_length, self.ls)
        #     padding = np.zeros(pad_shape, dtype=np.float32)
        #     object_label_array = np.concatenate([object_label_array, padding], axis=0)

        object_feature_array = object_feature_array.transpose(1, 0, 2) # (observe_len, num_visible_object, fs)
        object_label_array = object_label_array.transpose(1, 0, 2) # (future_len, num_visible_object, ls)

        # normalization (x, y만 rescale)
        object_feature_array[:, :, 0:1] /= rescale_x
        object_feature_array[:, :, 1:2] /= rescale_y

        object_label_array[:, :, 0:1] /= rescale_x
        object_label_array[:, :, 1:2] /= rescale_y


        # if perturbation is not None:
        #     for _obj_id in perturbation["ready_value"]:
        #         ori_data[0,3:5,:self.obs_length,obj_index[_obj_id]] += torch.transpose(perturbation["ready_value"][_obj_id], 0, 1)

        _input_data = torch.tensor(object_feature_array, dtype=torch.float32).to(self.dev)
        output_loc_GT = torch.tensor(object_label_array, dtype=torch.float32).to(self.dev)
                    
        return _input_data, output_loc_GT, rescale_xy, obj_index

    def postprocess(self, input_data, perturbation, predicted, rescale_xy, obj_index):
        predicted = predicted * rescale_xy

        if perturbation is not None:
            for _obj_id in perturbation["ready_value"]:
                input_data["objects"][str(_obj_id)]["perturbation"] = perturbation["ready_value"][_obj_id].detach().cpu().numpy()
                input_data["objects"][str(_obj_id)]["observe_trace"] += input_data["objects"][str(_obj_id)]["perturbation"]
            
            if "loss" in perturbation and perturbation["loss"] is not None:
                observe_traces = {}
                future_traces = {}
                predict_traces = {}

                for _obj_id, obj in input_data["objects"].items():
                    if not obj["visible"]:
                        continue

                    observe_traces[_obj_id] = torch.from_numpy(input_data["objects"][str(_obj_id)]["observe_trace"]).cuda()
                    future_traces[_obj_id] = torch.from_numpy(input_data["objects"][str(_obj_id)]["future_trace"]).cuda()
                    #predict_traces[_obj_id] = torch.from_numpy(predicted[0,:,obj_index[_obj_id],:]).cuda()
                    predict_traces[_obj_id] = torch.transpose(predicted[0,:,obj_index[_obj_id],:], 0, 1)
                
                if "attack_opts" in perturbation:
                    attack_opts = perturbation["attack_opts"]
                else:
                    attack_opts = None
                loss = perturbation["loss"](observe_traces, future_traces, predict_traces, perturbation["obj_id"], perturbation["ready_value"][perturbation["obj_id"]], **attack_opts)
            else:
                loss = None
        else:
            loss = None

        for obj_id, obj in input_data["objects"].items():
            if not obj["visible"]:
                continue
            
            obj["predict_trace"] = predicted[0, : , obj_index[obj_id], :].cpu().detach().numpy()

        return input_data, loss