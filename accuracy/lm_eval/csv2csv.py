import pandas as pd
import ast
from itertools import combinations

# --- Jaccard 相似度计算函数 ---
def calculate_jaccard_similarity(list1, list2):
    """计算两个列表（作为集合）之间的 Jaccard 相似度"""
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 1.0
    
    return intersection / union

# --- 主逻辑 ---
# 1. 读取原始日志文件
try:
    df_raw = pd.read_csv('prefetch_log.csv')
except FileNotFoundError:
    print("错误：请确保 'prefetch_log.csv' 文件与脚本在同一目录下。")
    exit()

# 2. 创建用于观察的透视表 (与上次相同)
df_raw['prefetched_indices_str'] = df_raw['prefetched_indices'].astype(str)
pivoted_df = df_raw.pivot_table(
    index=['request_id', 'token_idx_in_prompt'],
    columns='layer_idx',
    values='prefetched_indices_str',
    aggfunc='first'
).reset_index()
pivoted_df.columns.name = None # 清理列索引名

# 3. 计算每行的平均相似度
print("正在为每个生成步骤计算平均层间相似度...")

similarity_scores = []
# 遍历透视表的每一行
for index, row in pivoted_df.iterrows():
    # 提取所有层的预取列表
    layer_indices = []
    # 从第2列开始是layer数据
    for col in pivoted_df.columns[2:]:
        # 获取单元格的值，如果为空(NaN)则使用空列表
        indices_str = row[col]
        if pd.notna(indices_str):
            try:
                # 将字符串形式的列表安全转回列表
                layer_indices.append(ast.literal_eval(indices_str))
            except (ValueError, SyntaxError):
                layer_indices.append([]) # 如果解析失败，视为空列表
        else:
            layer_indices.append([])

    # 计算该行所有层两两之间的 Jaccard 相似度
    if len(layer_indices) < 2:
        similarity_scores.append(1.0) # 如果层数少于2，无法比较，相似度为1
        continue

    pair_similarities = []
    for list1, list2 in combinations(layer_indices, 2):
        pair_similarities.append(calculate_jaccard_similarity(list1, list2))
    
    # 计算平均值
    if not pair_similarities:
        avg_similarity = 1.0 # 如果没有可比较的对，则为1
    else:
        avg_similarity = sum(pair_similarities) / len(pair_similarities)
        
    similarity_scores.append(avg_similarity)

# 4. 将平均相似度作为新的一列添加到 DataFrame
pivoted_df['avg_jaccard_similarity'] = similarity_scores

# 5. 保存结果
output_filename = 'prefetch_view_with_similarity.csv'
pivoted_df.to_csv(output_filename, index=False)

print(f"\n带有平均相似度得分的表格已保存到 '{output_filename}'")
print("\n新表格预览:")
# 调整打印的列，使其更易读
preview_cols = list(pivoted_df.columns[:5]) + ['...', 'avg_jaccard_similarity']
print(pivoted_df.head(15).to_string(
    columns=pivoted_df.columns if len(pivoted_df.columns) < 8 else preview_cols, 
    float_format="%.4f"
))