#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
给建筑能耗 CSV 数据集添加噪声。

默认读取 /home/ykj/build/data 下的所有 CSV 文件，自动识别数值列并添加高斯噪声，
输出到 /home/ykj/build/data_noise。时间戳等非数值列会原样保留。

示例：
  python 数据加噪声.py
  python 数据加噪声.py --columns electricity --noise-level 0.05
  python 数据加噪声.py --noise-type uniform --noise-level 0.1 --clip-nonnegative
"""

import argparse
import csv
import random
from pathlib import Path


DEFAULT_INPUT_DIR = Path("/home/ykj/build/data")
DEFAULT_OUTPUT_DIR = Path("/home/ykj/build/data_noise")


def parse_args():
    parser = argparse.ArgumentParser(
        description="给建筑能耗 CSV 数据集添加噪声，并保存为新的 CSV 文件。"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"原始数据目录，默认：{DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"加噪后数据输出目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="需要加噪的列名；不填则自动对所有数值列加噪。",
    )
    parser.add_argument(
        "--exclude-columns",
        nargs="+",
        default=["timestamp", "date", "datetime", "time"],
        help="自动识别数值列时排除的列名，默认排除常见时间列。",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.5,
        help="噪声强度。高斯噪声为 列标准差 * noise-level，均匀噪声为 列标准差 * noise-level 范围内随机扰动。默认：0.05",
    )
    parser.add_argument(
        "--noise-type",
        choices=["gaussian", "uniform"],
        default="gaussian",
        help="噪声类型，默认：gaussian。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认：42。",
    )
    parser.add_argument(
        "--clip-nonnegative",
        action="store_true",
        help="将加噪后的负数截断为 0，适合 electricity 等不能为负的列。",
    )
    parser.add_argument(
        "--suffix",
        default="_noise",
        help="输出文件名后缀，默认：_noise。",
    )
    return parser.parse_args()


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_numeric_column(rows, column):
    values = [to_float(row.get(column, "")) for row in rows if row.get(column, "") != ""]
    return bool(values) and all(value is not None for value in values)


def choose_noise_columns(fieldnames, rows, requested_columns, excluded_columns):
    if requested_columns:
        missing = [column for column in requested_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"找不到指定列：{', '.join(missing)}")
        non_numeric = [
            column for column in requested_columns if not is_numeric_column(rows, column)
        ]
        if non_numeric:
            raise ValueError(f"指定列不是纯数值列：{', '.join(non_numeric)}")
        return requested_columns

    excluded = {column.lower() for column in excluded_columns}
    return [
        column
        for column in fieldnames
        if column.lower() not in excluded and is_numeric_column(rows, column)
    ]


def column_std(rows, column):
    values = [to_float(row[column]) for row in rows if to_float(row.get(column, "")) is not None]
    if len(values) < 2:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def format_like_original(original_value, new_value):
    if "." not in original_value:
        return str(int(round(new_value)))

    decimals = len(original_value.rstrip("0").split(".")[-1])
    decimals = max(decimals, 1)
    return f"{new_value:.{decimals}f}"


def add_noise(rows, columns, noise_level, noise_type, clip_nonnegative):
    scales = {column: column_std(rows, column) * noise_level for column in columns}
    noisy_rows = []

    for row in rows:
        new_row = dict(row)
        for column in columns:
            value = to_float(row.get(column, ""))
            if value is None:
                continue

            scale = scales[column]
            if noise_type == "gaussian":
                noisy_value = value + random.gauss(0.0, scale)
            else:
                noisy_value = value + random.uniform(-scale, scale)

            if clip_nonnegative:
                noisy_value = max(0.0, noisy_value)

            new_row[column] = format_like_original(row[column], noisy_value)
        noisy_rows.append(new_row)

    return noisy_rows, scales


def output_path_for(input_path, input_dir, output_dir, suffix):
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.with_name(f"{relative_path.stem}{suffix}{relative_path.suffix}")


def process_file(input_path, input_dir, output_dir, args):
    fieldnames, rows = read_csv(input_path)
    if not rows:
        print(f"跳过空文件：{input_path}")
        return

    columns = choose_noise_columns(
        fieldnames, rows, args.columns, args.exclude_columns
    )
    if not columns:
        print(f"未发现可加噪的数值列，跳过：{input_path}")
        return

    noisy_rows, scales = add_noise(
        rows,
        columns,
        args.noise_level,
        args.noise_type,
        args.clip_nonnegative,
    )
    output_path = output_path_for(input_path, input_dir, output_dir, args.suffix)
    write_csv(output_path, fieldnames, noisy_rows)

    scale_text = ", ".join(f"{column}={scales[column]:.6g}" for column in columns)
    print(f"完成：{input_path} -> {output_path}")
    print(f"加噪列：{', '.join(columns)}")
    print(f"噪声标准/范围：{scale_text}")


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.noise_level < 0:
        raise ValueError("--noise-level 不能为负数")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{args.input_dir}")

    csv_files = sorted(args.input_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"目录中没有 CSV 文件：{args.input_dir}")

    for csv_file in csv_files:
        process_file(csv_file, args.input_dir, args.output_dir, args)


if __name__ == "__main__":
    main()
