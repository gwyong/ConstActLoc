import os, re, json
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

def load_prediction_json(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    try:
        data = json.loads(txt)
        return pd.DataFrame(data if isinstance(data, list) else [data])
    except json.JSONDecodeError:
        return pd.DataFrame([json.loads(line) for line in txt.splitlines() if line.strip()])


def normalize_action(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def parse_frame_path(frame_path):
    name = os.path.basename(frame_path.replace("\\", "/"))
    stem = os.path.splitext(name)[0]

    # last two numbers: second, frame_index
    m = re.search(r"_(\d+)_(\d+)$", stem)
    if m is None:
        raise ValueError(f"Cannot parse frame_path: {frame_path}")

    return stem[:m.start()], int(m.group(1)), int(m.group(2))

def collapse_sequence(seq):
    collapsed = []
    for x in seq:
        if x is None:
            continue
        if len(collapsed) == 0 or collapsed[-1] != x:
            collapsed.append(x)
    return collapsed


def levenshtein_distance(a, b):
    # a, b: list of action labels
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[-1][-1]

def evaluate_actions(
    pred_json_path,
    gt_csv_path,
    output_txt_path="evaluation_results.txt",
    top_n=50,
):
    pred_df = load_prediction_json(pred_json_path)
    gt_df = pd.read_csv(gt_csv_path)

    parsed = pred_df["frame_path"].apply(parse_frame_path)
    pred_df[["base_filename", "second", "frame_idx"]] = pd.DataFrame(
        parsed.tolist(), index=pred_df.index
    )

    pred_df["pred_action"] = pred_df["action"].apply(normalize_action)

    for c in ["action_1", "action_2", "action_3"]:
        gt_df[c] = gt_df[c].apply(normalize_action)

    merged = pred_df.merge(
        gt_df,
        on=["base_filename", "second"],
        how="left",
        validate="one_to_one",
    )


    def clean_action_text(x):
        if pd.isna(x):
            return None

        x = str(x).strip().lower()
        x = x.replace("\\", "")   # remove backslash
        return x
    
    merged["pred_action"] = merged["pred_action"].apply(clean_action_text)
    
    def gt_actions(row):
        actions = set()

        for col in ["action_1", "action_2", "action_3"]:
            a = clean_action_text(row[col])
            if a is not None:
                actions.add(a)

        return actions
    
    merged["gt_actions"] = merged.apply(gt_actions, axis=1)

    # exact accuracy
    merged["correct"] = merged.apply(
        lambda r: int(r["pred_action"] in r["gt_actions"]),
        axis=1,
    )

    accuracy = merged["correct"].mean()
    total_cost = merged["inference_cost"].sum()
    total_time = merged["inference_time"].sum()
    avg_cost = merged["inference_cost"].mean()
    avg_time = merged["inference_time"].mean()

    # 1-second overlap accuracy
    gt_lookup = {
        (r.base_filename, r.second): {
            a for a in [r.action_1, r.action_2, r.action_3]
            if a is not None
        }
        for r in gt_df.itertuples(index=False)
    }

    def overlap_correct(row):
        candidate_actions = set()
        for s in [row["second"] - 1, row["second"], row["second"] + 1]:
            candidate_actions |= gt_lookup.get((row["base_filename"], s), set())
        return int(row["pred_action"] in candidate_actions)

    merged["overlap_correct"] = merged.apply(overlap_correct, axis=1)
    overlap_accuracy = merged["overlap_correct"].mean()

        # foreground-only accuracy (exclude GT containing only 'none')
    merged["is_foreground"] = merged["gt_actions"].apply(
        lambda x: not (len(x) == 1 and "none" in x)
    )

    foreground_df = merged[merged["is_foreground"]].copy()
    foreground_accuracy = foreground_df["correct"].mean()

    # per-class metrics
    # For multi-GT labels, if prediction matches any GT -> assign predicted label as true label
    # Otherwise assign primary GT label (action_1) for confusion accounting
    eval_true = []
    eval_pred = []

    for _, row in foreground_df.iterrows():
        pred = row["pred_action"]

        if pred in row["gt_actions"]:
            true_label = pred
        else:
            true_label = row["action_1"]

        eval_true.append(true_label)
        eval_pred.append(pred)

    all_labels = sorted(set(eval_true) | set(eval_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        eval_true,
        eval_pred,
        labels=all_labels,
        zero_division=0,
    )
    macro_f1 = f1.mean()
    per_class_df = pd.DataFrame({
        "class": all_labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).sort_values("support", ascending=False)
    # edit distance analysis
    edit_distance_rows = []

    for video_name, video_df in merged.groupby("base_filename"):
        video_df = video_df.sort_values("second")

        pred_seq = collapse_sequence(video_df["pred_action"].tolist())

        # Use action_1 as the primary GT sequence
        gt_seq = collapse_sequence(video_df["action_1"].apply(clean_action_text).tolist())

        edit_dist = levenshtein_distance(pred_seq, gt_seq)

        # normalized edit distance
        denom = max(len(pred_seq), len(gt_seq), 1)
        norm_edit_dist = edit_dist / denom

        edit_distance_rows.append({
            "base_filename": video_name,
            "gt_sequence_length": len(gt_seq),
            "pred_sequence_length": len(pred_seq),
            "edit_distance": edit_dist,
            "normalized_edit_distance": norm_edit_dist,
            "gt_sequence": gt_seq,
            "pred_sequence": pred_seq,
        })

    edit_distance_df = pd.DataFrame(edit_distance_rows)

    avg_edit_distance = edit_distance_df["edit_distance"].mean()
    avg_normalized_edit_distance = edit_distance_df["normalized_edit_distance"].mean()
    
    # missed class analysis
    missed_counter = Counter()
    confusion_counter = Counter()

    wrong_df = merged[merged["correct"] == 0].copy()

    for _, row in wrong_df.iterrows():
        for gt_action in row["gt_actions"]:
            missed_counter[gt_action] += 1
            confusion_counter[(gt_action, row["pred_action"])] += 1

    gt_class_counter = Counter()
    for _, row in merged.iterrows():
        for gt_action in row["gt_actions"]:
            gt_class_counter[gt_action] += 1

    miss_rate_rows = []
    for cls, miss_count in missed_counter.items():
        total_count = gt_class_counter[cls]
        miss_rate_rows.append((cls, miss_count, total_count, miss_count / total_count))

    miss_rate_df = pd.DataFrame(
        miss_rate_rows,
        columns=["gt_class", "miss_count", "total_count", "miss_rate"],
    ).sort_values("miss_rate", ascending=False)

    confusion_df = pd.DataFrame(
        [
            (gt, pred, count)
            for (gt, pred), count in confusion_counter.items()
        ],
        columns=["gt_class", "predicted_as", "count"],
    ).sort_values("count", ascending=False)

    result_text = []
    result_text.append("=== Overall Results ===")
    result_text.append(f"Accuracy: {accuracy:.4f}")
    result_text.append(f"Foreground Accuracy (exclude none-only GT): {foreground_accuracy:.4f}")
    result_text.append(f"Overlap Accuracy (+/- 1 sec): {overlap_accuracy:.4f}")
    result_text.append(f"Macro F1 (foreground only): {macro_f1:.4f}")
    result_text.append(f"Total Cost: {total_cost:.4f}")
    result_text.append(f"Total Inference Time: {total_time:.4f}")
    result_text.append(f"Average Cost per Frame: {avg_cost:.4f}")
    result_text.append(f"Average Inference Time per Frame: {avg_time:.4f}")
    result_text.append(f"Average Edit Distance per Video: {avg_edit_distance:.4f}")
    result_text.append(f"Average Normalized Edit Distance per Video: {avg_normalized_edit_distance:.4f}")
    result_text.append("")
    result_text.append("=== Most Missed Classes by Miss Rate ===")
    result_text.append(miss_rate_df.head(top_n).to_string(index=False))
    result_text.append("")
    result_text.append("=== Most Frequent Confusions ===")
    result_text.append(confusion_df.head(top_n).to_string(index=False))
    result_text.append("")
    result_text.append("=== Per-Class Precision / Recall / F1 ===")
    result_text.append(per_class_df.to_string(index=False))
    result_text = "\n".join(result_text)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(result_text)

    return {
        "accuracy": f"{accuracy:.4f}",
        "overlap_accuracy": f"{overlap_accuracy:.4f}",
        "total_cost": f"{total_cost:.4f}",
        "total_inference_time": f"{total_time:.4f}",
        "average_cost": f"{avg_cost:.4f}",
        "average_inference_time": f"{avg_time:.4f}",
        "average_edit_distance": f"{avg_edit_distance:.4f}",
        "average_normalized_edit_distance": f"{avg_normalized_edit_distance:.4f}",
        "edit_distance_df": edit_distance_df,
        "merged_df": merged,
        "miss_rate_df": miss_rate_df,
        "confusion_df": confusion_df,
        "foreground_accuracy": f"{foreground_accuracy:.4f}",
        "per_class_df": per_class_df,
    }


if __name__ == "__main__":
    # pred_json_path = "output/inference_results_gpt-5.4.json"
    # pred_json_path = "output/inference_results_claude-sonnet-4-6.json"
    # pred_json_path = "output/inference_results_gemini-3.1-pro-preview_video_gpt_format.json"
    # pred_json_path = "output/inference_results_gpt-5.4-mini.json"
    # pred_json_path = "output/inference_results_claude-haiku-4-5-20251001.json"
    pred_json_path = "output/inference_results_gemini-3.1-flash-lite-preview_video_gpt_format.json"
    
    model_name = os.path.basename(pred_json_path).replace("inference_results_", "").replace(".json", "")
    gt_csv_path = "data/GT_fully_annotated.csv"
    results = evaluate_actions(
        pred_json_path=pred_json_path,
        gt_csv_path=gt_csv_path,
        output_txt_path=f"output/evaluation_results_{model_name}.txt",
    )
    results["confusion_df"].to_csv(
    f"output/confusion/confusion_{model_name}.csv",
    index=False
    )
"""
GPT-5.4
Accuracy: 0.7819
Foreground Accuracy (exclude none-only GT): 0.8266
Overlap Accuracy (+/- 1 sec): 0.8343
Macro F1 (foreground only): 0.8030
Total Cost: 22.0406
Total Inference Time: 7707.7766
Average Cost per Frame: 0.0051
Average Inference Time per Frame: 1.7888

GPT-5.4 Mini
Accuracy: 0.6542
Foreground Accuracy (exclude none-only GT): 0.6519
Overlap Accuracy (+/- 1 sec): 0.7081
Macro F1 (foreground only): 0.6388
Total Cost: 6.6836
Total Inference Time: 5287.3105
Average Cost per Frame: 0.0016
Average Inference Time per Frame: 1.2270

Claude Sonnet 4-6
Accuracy: 0.7424
Foreground Accuracy (exclude none-only GT): 0.8646
Overlap Accuracy (+/- 1 sec): 0.8009
Macro F1 (foreground only): 0.8021
Total Cost: 26.2915
Total Inference Time: 9239.3518
Average Cost per Frame: 0.0061
Average Inference Time per Frame: 2.1442

Claude Haiku 4-5-20251001
Accuracy: 0.5570
Foreground Accuracy (exclude none-only GT): 0.6326
Overlap Accuracy (+/- 1 sec): 0.6073
Macro F1 (foreground only): 0.5451
Total Cost: 8.9643
Total Inference Time: 6194.7831
Average Cost per Frame: 0.0021
Average Inference Time per Frame: 1.4376
Average Edit Distance per Video: 11.5800
Average Normalized Edit Distance per Video: 0.6907

Gemini 3.1 Pro Preview (video format, converted to frame-level for evaluation)
Accuracy: 0.7793
Foreground Accuracy (exclude none-only GT): 0.8250
Overlap Accuracy (+/- 1 sec): 0.8459
Macro F1 (foreground only): 0.8123
Total Cost: 0.8335
Total Inference Time: 1623.1963
Average Cost per Frame: 0.0002
Average Inference Time per Frame: 0.3767

Gemini 3.1 Flash Lite Preview (video format, converted to frame-level for evaluation)
Accuracy: 0.7517
Foreground Accuracy (exclude none-only GT): 0.8453
Overlap Accuracy (+/- 1 sec): 0.8185
Macro F1 (foreground only): 0.8409
Total Cost: 0.1044
Total Inference Time: 532.7031
Average Cost per Frame: 0.0000
Average Inference Time per Frame: 0.1236
Average Edit Distance per Video: 3.2100
Average Normalized Edit Distance per Video: 0.3816
"""