import os, re, json
import pandas as pd
from collections import Counter, defaultdict
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
    x = str(x).strip().lower()
    x = x.replace("\\", "")   # remove backslash
    return x


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
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
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
    merged["pred_action"] = merged["pred_action"].apply(normalize_action)
    
    def gt_actions(row):
        actions = set()

        for col in ["action_1", "action_2", "action_3"]:
            a = normalize_action(row[col])
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
    one_min_cost = avg_cost * 60
    one_min_time = avg_time * 60

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
    result_text.append(f"Estimated Inference Time per Minute: {one_min_time:.4f}")
    result_text.append(f"Estimated Cost per Minute: {one_min_cost:.4f}")
    result_text = "\n".join(result_text)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(result_text)

    return {
        "accuracy": f"{accuracy:.4f}",
        "overlap_accuracy": f"{overlap_accuracy:.4f}",
        "foreground_accuracy": f"{foreground_accuracy:.4f}",
        "macro_f1": f"{macro_f1:.4f}",
        "total_cost": f"{total_cost:.4f}",
        "total_inference_time": f"{total_time:.4f}",
        "average_cost": f"{avg_cost:.4f}",
        "average_inference_time": f"{avg_time:.4f}",
        "estimated_cost_per_minute": f"{one_min_cost:.4f}",
        "estimated_inference_time_per_minute": f"{one_min_time:.4f}",
    }


def compute_aggregated_recall(
    pred_json_paths,
    gt_csv_path,
    top_n=10,
    output_csv_path=None,
):
    gt_df = pd.read_csv(gt_csv_path)

    for c in ["action_1", "action_2", "action_3"]:
        gt_df[c] = gt_df[c].apply(normalize_action)

    total_gt_counter = Counter()
    correct_counter = Counter()
    wrong_pred_counter = defaultdict(Counter)

    for pred_json_path in pred_json_paths:
        pred_df = load_prediction_json(pred_json_path)

        parsed = pred_df["frame_path"].apply(parse_frame_path)
        pred_df[["base_filename", "second", "frame_idx"]] = pd.DataFrame(
            parsed.tolist(), index=pred_df.index
        )

        pred_df["pred_action"] = pred_df["action"].apply(normalize_action)

        merged = pred_df.merge(
            gt_df,
            on=["base_filename", "second"],
            how="left",
            validate="one_to_one",
        )

        def get_gt_actions(row):
            actions = set()
            for c in ["action_1", "action_2", "action_3"]:
                a = normalize_action(row[c])
                if a is not None:
                    actions.add(a)
            return actions

        merged["gt_actions"] = merged.apply(get_gt_actions, axis=1)

        for _, row in merged.iterrows():
            pred = row["pred_action"]
            gt_actions = row["gt_actions"]

            if len(gt_actions) == 0:
                continue

            # Case 1: prediction matches one of multiple GT labels
            # Treat the frame as correctly detected.
            # Only the matched GT class is counted.
            if pred in gt_actions:
                total_gt_counter[pred] += 1
                correct_counter[pred] += 1

            # Case 2: prediction does not match any GT label
            # Then all GT labels in that frame are considered missed.
            else:
                for gt in gt_actions:
                    total_gt_counter[gt] += 1
                    wrong_pred_counter[gt][pred] += 1

    rows = []
    for gt_class, total_gt in total_gt_counter.items():
        correct = correct_counter[gt_class]
        missed = total_gt - correct
        recall = correct / total_gt if total_gt > 0 else 0

        if len(wrong_pred_counter[gt_class]) > 0:
            filtered_counter = Counter({
                k: v
                for k, v in wrong_pred_counter[gt_class].items()
                if k != "none"
            })

            if len(filtered_counter) > 0:
                top_wrong_class, top_wrong_count = filtered_counter.most_common(1)[0]
            else:
                top_wrong_class, top_wrong_count = None, 0

        else:
            top_wrong_class, top_wrong_count = None, 0
        
        none_miss_count = wrong_pred_counter[gt_class].get("none", 0)
        rows.append({
            "gt_class": gt_class,
            "total_count": total_gt,
            "correct_count": correct,
            "missed_count": missed,
            "detection_rate_recall": recall,
            "detection_rate_recall_percent": recall * 100,
            "most_common_wrong_prediction": top_wrong_class,
            "most_common_wrong_prediction_count": top_wrong_count,
            "none_miss_count": none_miss_count,
        })

    result_df = pd.DataFrame(rows)

    result_df = result_df.sort_values(
        ["detection_rate_recall", "missed_count"],
        ascending=[True, False]
    ).reset_index(drop=True)

    result_df.insert(0, "rank", result_df.index + 1)

    if output_csv_path is not None:
        result_df.to_csv(output_csv_path, index=False)

    return result_df.head(top_n), result_df

if __name__ == "__main__":
    # pred_json_path = "output/inference_results_gpt-5.4.json"
    # pred_json_path = "output/inference_results_gpt-5.4-mini.json"
    # pred_json_path = "output/inference_results_claude-sonnet-4-6.json"
    # pred_json_path = "output/inference_results_claude-haiku-4-5-20251001.json"
    # pred_json_path = "output/inference_results_gemini-3.1-pro-preview_video_gpt_format.json"
    pred_json_path = "output/inference_results_gemini-3.1-flash-lite-preview_video_gpt_format.json"
    
    model_name = os.path.basename(pred_json_path).replace("inference_results_", "").replace(".json", "")
    gt_csv_path = "data/GT_fully_annotated.csv"
    results = evaluate_actions(
        pred_json_path=pred_json_path,
        gt_csv_path=gt_csv_path,
        output_txt_path=f"output/evaluation_results_{model_name}.txt",
    )
    # results["confusion_df"].to_csv(
    # f"output/confusion/confusion_{model_name}.csv",
    # index=False
    # )

    # pred_json_paths = [
    # "output/inference_results_gpt-5.4.json",
    # "output/inference_results_gpt-5.4-mini.json",
    # "output/inference_results_claude-sonnet-4-6.json",
    # "output/inference_results_claude-haiku-4-5-20251001.json",
    # "output/inference_results_gemini-3.1-pro-preview_video_gpt_format.json",
    # "output/inference_results_gemini-3.1-flash-lite-preview_video_gpt_format.json",
    # ]

    # top10_df, full_df = compute_aggregated_recall(
    #     pred_json_paths=pred_json_paths,
    #     gt_csv_path=gt_csv_path,
    #     top_n=10,
    #     output_csv_path="output/aggregated_recall.csv",
    # )

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