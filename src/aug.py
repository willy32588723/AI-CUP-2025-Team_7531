import os
import numpy as np
import pandas as pd
from collections import Counter

# 路徑與參數設定
CSV_PATH = "train_info.csv"
TXT_DIR = "train_data"
OUTPUT_DIR = "train_data_augmented"
AUGMENT_CSV = "train_info_augmented.csv"

AUG_TIMES = 3           # 每筆擴增幾次
NOISE_STD = 0.05        # 噪音標準差
START_ID = 3129         # 從哪個數字開始命名新 unique_id.txt

os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)

# 統計 play years 、level類別，找最少的類別
class_counts = Counter(df["level"])
min_count = min(class_counts.values())
minority_classes = [cls for cls, cnt in class_counts.items() if cnt == min_count]
print(f"少數類別: {minority_classes} (數量: {min_count})")

# 開始擴增
aug_rows = []
new_id = START_ID
for idx, row in df.iterrows():
    if row["level"] not in minority_classes:
        continue

    original_id = row["unique_id"]
    original_txt_path = os.path.join(TXT_DIR, f"{original_id}.txt")

    if not os.path.exists(original_txt_path):
        print(f"缺少檔案: {original_txt_path}")
        continue

    data = np.loadtxt(original_txt_path)

    for _ in range(AUG_TIMES):
        new_uid = str(new_id)
        new_txt_path = os.path.join(OUTPUT_DIR, f"{new_uid}.txt")

        noisy_data = data + np.random.normal(0, NOISE_STD, data.shape)
        np.savetxt(new_txt_path, noisy_data, fmt="%.6f")

        new_row = row.copy()
        new_row["unique_id"] = new_uid
        aug_rows.append(new_row)

        new_id += 1

# 合併後儲存
aug_df = pd.DataFrame(aug_rows)
final_df = pd.concat([df, aug_df], ignore_index=True)
final_df.to_csv(AUGMENT_CSV, index=False)

print(f"已針對少數類別擴增 {len(aug_rows)} 筆樣本")
print(f"所有新增 .txt 已儲存至：{OUTPUT_DIR}")
print(f"更新後的 train_info 已儲存為：{AUGMENT_CSV}")
