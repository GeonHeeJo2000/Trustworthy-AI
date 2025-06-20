import os, json, random
from tqdm import tqdm

def build_sample_file(data_dir: str, out_path: str = "sample.txt", seed=42):
    if seed is not None:
        random.seed(seed)

    lines = []
    files = os.listdir(data_dir)
    selected_files = random.sample(files, min(1000, len(files)))

    for fname in tqdm(selected_files):
        frame_idx = os.path.splitext(fname)[0] 
        with open(os.path.join(data_dir, fname), "r") as f:
            frame = json.load(f)

        # visible 선수 ID(문자열) 모으기
        visibles = [oid for oid, obj in frame["objects"].items() if obj["visible"]]

        pick = random.choice(visibles)           # 하나 무작위 선택
        lines.append(f"{frame_idx} {pick}")

    # sample.txt 저장
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"saved {len(lines)} lines → {out_path}")

if __name__ == "__main__":
    build_sample_file(
        data_dir="./data/dataset/dfl/multi_frame/raw",
        out_path="./data/dataset/dfl/multi_frame/samples.txt",
        seed=42                   # (옵션) 항상 같은 결과를 원하면 시드 고정
    )
