import json
import os
import random


def main():
    # 设置
    save_train_file_name = (
        "train_val_dataset/train_start_and_end_eye_head_pose_dataset.json"
    )
    save_val_file_name = (
        "train_val_dataset/val_start_and_end_eye_head_pose_dataset.json"
    )
    train_ratio = 0.8

    # 加载并整合所有json文件中的数据
    all_files_name = os.listdir(".")
    json_files = []
    for file_name in all_files_name:
        if file_name.endswith(".json"):
            json_files.append(file_name)
    json_files = sorted(json_files, key=lambda x: x.split(".")[0])
    all_data_dict = {}
    for json_file_name in json_files:
        with open(json_file_name, "r", encoding="utf-8") as f:
            json_file = json.load(f)
            all_data_dict.update(json_file)

    # 打乱数据并划分为训练集和验证集
    all_data_items = list(all_data_dict.items())
    random.shuffle(all_data_items)
    split_index = int(len(all_data_items) * train_ratio)
    train_data_items = all_data_items[:split_index]
    val_data_items = all_data_items[split_index:]
    train_data_dict = dict(train_data_items)
    val_data_dict = dict(val_data_items)

    # save train and val dataset json file
    with open(
        save_train_file_name,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(train_data_dict, f, ensure_ascii=False, indent=2)
    with open(
        save_val_file_name,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(val_data_dict, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
