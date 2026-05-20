import pandas as pd
import pickle

# ============================
# 1. 读取 MegaScale CSV
# ============================
csv_path = "../data_all/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
df = pd.read_csv(csv_path)

# 清洗（与 SPURS/ProStab 一致）
df = df[df.ddG_ML != "-"].reset_index(drop=True)
df = df[~df.mut_type.str.contains("ins|del|:")].reset_index(drop=True)

print("原始 CSV 行数:", len(df))
print("原始 WT 数:", df["WT_name"].nunique())

# ============================
# 2. 读取 m8 文件，收集 sseqid
# ============================
m8_path = "mmseq_mut_search_0.25.m8"
remove_idx = set()

with open(m8_path) as f:
    for line in f:
        sseqid = int(line.split("\t")[1])
        remove_idx.add(sseqid)

print("m8 中 unique sseqid 数:", len(remove_idx))

# ============================
# 3. 找到这些 index 对应的 WT_name
# ============================
remove_wt_names = set()

for idx in remove_idx:
    if idx < len(df):  # 防止越界
        remove_wt_names.add(df.loc[idx, "WT_name"])

print("需要删除的 WT_name 数:", len(remove_wt_names))

# ============================
# 4. 删除所有 WT_name 匹配的行（WT-level 过滤）
# ============================
before_rows = len(df)
before_wt = df["WT_name"].nunique()

df_filtered = df[~df["WT_name"].isin(remove_wt_names)].reset_index(drop=True)

after_rows = len(df_filtered)
after_wt = df_filtered["WT_name"].nunique()

print("\n===== 过滤结果（WT-level）=====")
print("过滤前 行数:", before_rows)
print("过滤后 行数:", after_rows)
print("删除 行数:", before_rows - after_rows)
print("过滤前 WT 数:", before_wt)
print("过滤后 WT 数:", after_wt)
print("删除 WT 数:", before_wt - after_wt)

# ============================
# 5. 加载 MegaScale 官方划分
# ============================
split_path = "mega_splits.pkl"
splits = pickle.load(open(split_path, "rb"))

train_wt = set(splits["train"])
val_wt   = set(splits["val"])
test_wt  = set(splits["test"])

# ============================
# 6. 按 WT_name 划分
# ============================
train_df = df_filtered[df_filtered["WT_name"].isin(train_wt)]
val_df   = df_filtered[df_filtered["WT_name"].isin(val_wt)]
test_df  = df_filtered[df_filtered["WT_name"].isin(test_wt)]

# ============================
# 7. 打印最终信息
# ============================
def print_info(name, df):
    print(f"\n===== {name} =====")
    print("WT 数:", df["WT_name"].nunique())
    print("突变行数:", len(df))
    if df["WT_name"].nunique() > 0:
        print("平均突变数/WT:", len(df) / df["WT_name"].nunique())

print_info("TRAIN", train_df)
print_info("VAL", val_df)
print_info("TEST", test_df)

# ============================
# 8. 保存结果
# ============================
train_df.to_csv("megascale_train_filtered.csv", index=False)
val_df.to_csv("megascale_val_filtered.csv", index=False)
test_df.to_csv("megascale_test_filtered.csv", index=False)

print("\n全部完成！")

