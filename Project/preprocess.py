import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import scipy.signal as signal
from scipy.ndimage import shift

data_path = './data/dataset/dfl'
save_path = './data/dataset/dfl/multi_frame/train'
os.makedirs(save_path, exist_ok=True)



def calc_player_accel(positions, window=7, polyorder=1, MAX_ACCEL = 8):
    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = 1 / 25 # 0.04 seconds

    # estimate velocities for players in team
    player_ids = [p[:3] for p in positions.columns if p.endswith("_x")]# and not p.startswith("B")]
    accel_data = {}
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = pd.Series([float(p) for p in positions[player+"_vx"]])
        vy = pd.Series([float(p) for p in positions[player+"_vy"]])

        ax = np.diff(vx, prepend=vx[0]) / dt
        ay = np.diff(vy, prepend=vy[0]) / dt
        
        accel = np.sqrt(vx**2 + vy**2)
        is_accel_outlier = accel > MAX_ACCEL
        is_outlier = is_accel_outlier | shift(is_accel_outlier, 1, cval=True)
        ax = pd.Series(np.where(is_outlier, np.nan, ax)).interpolate(limit_direction="both").values
        ay = pd.Series(np.where(is_outlier, np.nan, ay)).interpolate(limit_direction="both").values
 
        ax = signal.savgol_filter(ax, window_length=window, polyorder=polyorder)
        ay = signal.savgol_filter(ay, window_length=window, polyorder=polyorder)

        # Store player speed in x, y direction, and total speed in the dictionary
        accel_data[player + "_ax"] = ax
        accel_data[player + "_ay"] = ay

    accel_df = pd.DataFrame(accel_data).round(2)

    return pd.concat([positions, accel_df], axis=1)

def build_soccer_dataset(data_df,
                         obs_sec=4.0, pred_sec=2.0,
                         fps=25,
                         stride=5,
                         slide_step=25):
    """
    data_df: DataFrame, 각 행에 모든 선수의 x,y,vx,vy,ax,ay 컬럼을 가짐.
    obs_sec: 관측 시간 (초), pred_sec: 예측 시간 (초)
    orig_fps: 원본 프레임레이트 (25)
    obs_stride: 관측 다운샘플 스트라이드 (5)
    pred_stride: 예측 다운샘플 스트라이드 (2)
    slide_step: 윈도우 이동 스텝 (행 단위)
    """
    # 원본 row 단위 window 길이
    obs_rows  = int(obs_sec  * fps)  # 4*25 = 100
    pred_rows = int(pred_sec * fps)  # 2*25 = 50

    # 다운샘플 후 frame 수
    obs_len  = obs_rows  // stride   # 100/5 = 20
    pred_len = pred_rows // stride  # 50/5 = 10

    samples = []
    # 모든 슬라이딩 윈도우에 대해 샘플 생성
    for _, group_period in data_df.groupby("period_id"):
        for start in tqdm(range(0, len(group_period) - (obs_rows + pred_rows) + 1, slide_step)):
            window = group_period.iloc[start : start + obs_rows + pred_rows: stride].reset_index(drop=True)

            sample = {
                "observe_length":  obs_len + pred_len,
                "predict_length": 0,
                "time_step":       stride / fps,    # 5/25 = 0.2s
                "feature_dimension": 6,
                "objects": {}
            }

            # 플레이어 ID 추출
            player_ids = sorted({col.split("_")[0] for col in window.columns if col.endswith("_x")})

            for pid in player_ids:
                x_all  = window[f"{pid}_x"].values.round(2)
                y_all  = window[f"{pid}_y"].values.round(2)
                vx_all  = window[f"{pid}_vx"].values.round(2)
                vy_all  = window[f"{pid}_vy"].values.round(2)
                ax_all  = window[f"{pid}_ax"].values.round(2)
                ay_all  = window[f"{pid}_ay"].values.round(2)

                observe_feature = np.vstack(
                    (x_all, y_all, 
                     vx_all, vy_all,
                     ax_all, ay_all)
                ).T

                sample["objects"][pid] = {
                    "type": 1,  # player
                    "complete": bool(not np.any(np.isnan(observe_feature))),
                    "visible":  bool(not np.any(np.isnan(observe_feature))),
                    "observe_trace":    np.vstack(
                        (x_all,  y_all)
                    ).T.tolist(),
                    "observe_feature": observe_feature.tolist(),
                    
                    # future정보는 추후에 observe_trace정보가 슬라이딩 과정을 통해 input & output으로 나누어짐
                    "observe_mask": [],
                    "future_trace": [],
                    "future_feature": [],
                    "predict_trace": [],
                    "future_mask": []
                }

            samples.append(sample)

    return samples

if __name__ == "__main__":
    idx = 0
    
    file_path = sorted(os.listdir(data_path))[:6] if "train" in save_path else sorted(os.listdir(data_path))[6:7]
    for folder_name in tqdm(file_path, desc=f"Processing {save_path}"):
        folder_path = os.path.join(data_path, folder_name)

        file_path = os.path.join(folder_path, 'positions.csv')

        data_df = pd.read_csv(file_path)
        a_cols = [col for col in data_df.columns if col.endswith("_ax") or col.endswith("_ay")]
        data_df = data_df.drop(columns=a_cols)
        data_df = calc_player_accel(data_df)

        samples = build_soccer_dataset(data_df)

        # JSON 파일로 저장
        for _, sample in enumerate(samples):
            if sum(o["visible"] for o in sample["objects"].values()) != 23:
                print("23명의 에이전트가 존재하지 않는다:", sum(o["visible"] for o in sample["objects"].values()))
                continue    

            filename = f"{idx}.json"
            filepath = os.path.join(save_path, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)
            idx += 1
