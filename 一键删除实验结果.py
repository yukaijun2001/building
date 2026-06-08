from pathlib import Path
import shutil


BASE_DIR = Path("/home/ykj/build/Time-Series-Library")

TARGET_DIRS = [
    BASE_DIR / "checkpoints",
    BASE_DIR / "results",
    BASE_DIR / "outputs",
    BASE_DIR / "test_results",
]

TARGET_FILES = [
    BASE_DIR / "result_long_term_forecast.txt",
]


def delete_dir(path: Path) -> None:
    if not path.exists():
        print(f"跳过：{path} 不存在")
        return

    if not path.is_dir():
        print(f"跳过：{path} 不是目录")
        return

    shutil.rmtree(path)
    print(f"已删除目录：{path}")


def delete_file(path: Path) -> None:
    if not path.exists():
        print(f"跳过：{path} 不存在")
        return

    if not path.is_file():
        print(f"跳过：{path} 不是文件")
        return

    path.unlink()
    print(f"已删除：{path}")


def main() -> None:
    for target_dir in TARGET_DIRS:
        delete_dir(target_dir)

    for target_file in TARGET_FILES:
        delete_file(target_file)


if __name__ == "__main__":
    main()
