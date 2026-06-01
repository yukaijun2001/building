from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent / "Time-Series-Library"
TARGET_DIRS = ("checkpoints", "results", "test_results")
KEYWORD = "PatchGatedLSTM"
PYCACHE_DIRNAME = "__pycache__"


def delete_matching_dirs() -> None:
    deleted_count = 0

    for dirname in TARGET_DIRS:
        parent_dir = PROJECT_ROOT / dirname
        if not parent_dir.exists():
            print(f"跳过：目录不存在 {parent_dir}")
            continue

        for child in sorted(parent_dir.iterdir()):
            if child.is_dir() and KEYWORD in child.name:
                shutil.rmtree(child)
                deleted_count += 1
                print(f"已删除：{child}")

    print(f"完成：共删除 {deleted_count} 个包含 {KEYWORD} 的文件夹。")


def delete_pycache_dirs() -> None:
    deleted_count = 0

    if not PROJECT_ROOT.exists():
        print(f"跳过：目录不存在 {PROJECT_ROOT}")
        return

    for pycache_dir in sorted(PROJECT_ROOT.rglob(PYCACHE_DIRNAME)):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            deleted_count += 1
            print(f"已删除：{pycache_dir}")

    print(f"完成：共删除 {deleted_count} 个 {PYCACHE_DIRNAME} 文件夹。")


if __name__ == "__main__":
    delete_matching_dirs()
    delete_pycache_dirs()
