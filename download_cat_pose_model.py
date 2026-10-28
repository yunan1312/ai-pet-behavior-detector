"""猫咪/动物姿态模型下载脚本（兼容入口）

推荐模型：RTMPose-M AP-10K 动物姿态模型（ONNX，猫/狗等 54 种动物）。
实际下载逻辑见 download_rtmpose.py。

使用方法：
    python download_cat_pose_model.py                  # 下载/校验推荐模型
    python download_cat_pose_model.py --model yolo11   # YOLO-pose 兼容回退
    python download_cat_pose_model.py --list           # 列出已有模型
"""

import argparse
import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "rtmpose": {
        "name": "RTMPose-M AP-10K",
        "description": "专用动物姿态模型，覆盖猫/狗等 54 种动物（推荐）",
        "filename": "rtmpose_ap10k.onnx",
        "urls": [],
        "size_mb": 52.0,
        "recommended": True,
    },
    "yolo11": {
        "name": "YOLO11n-Pose",
        "description": "COCO 人体关键点模型，仅在缺少动物姿态模型时兼容回退",
        "filename": "yolo11n-pose.pt",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
        ],
        "size_mb": 6.0,
        "recommended": False,
    },
    "yolo8": {
        "name": "YOLOv8n-Pose",
        "description": "COCO 人体关键点模型，兼容性好但非动物专用",
        "filename": "yolov8n-pose.pt",
        "urls": [
            "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n-pose.pt",
        ],
        "size_mb": 6.5,
        "recommended": False,
    },
}


def download_file(url: str, save_path: str, description: str = "") -> bool:
    """下载文件，显示进度"""
    print(f"\n  下载: {description}")
    print(f"  URL: {url}")
    print(f"  保存: {save_path}")

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "PetBehaviorAnalyzer/2.1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            total_size = int(response.headers.get("content-length", 0) or 0)
            if total_size > 0:
                print(f"  大小: {total_size / (1024 * 1024):.1f} MB")

            downloaded = 0
            with open(save_path, "wb") as handle:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(
                            f"\r  下载进度: {percent:.1f}% "
                            f"({downloaded / (1024 * 1024):.1f} / "
                            f"{total_size / (1024 * 1024):.1f} MB)"
                        )
                        sys.stdout.flush()

        print("\n  下载完成!")
        return True
    except Exception as error:
        print(f"\n  下载失败: {error}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def list_models():
    """列出已有模型文件"""
    print("\n" + "=" * 60)
    print("  已有模型文件")
    print("=" * 60)

    found = False
    for directory, suffix in ((MODELS_DIR, ""), (PROJECT_ROOT, " [根目录]")):
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and name.endswith((".onnx", ".pt", ".pth")):
                size = os.path.getsize(path) / (1024 * 1024)
                print(f"  - {name} ({size:.1f} MB){suffix}")
                found = True

    if not found:
        print("  未找到任何模型文件")

    print("=" * 60)


def download_model(model_key: str) -> bool:
    """下载指定模型"""
    config = MODEL_CONFIGS[model_key]

    # RTMPose 走专用下载脚本（含解压与 onnxruntime 校验）
    if model_key == "rtmpose":
        import download_rtmpose

        return download_rtmpose.main() == 0

    save_path = os.path.join(MODELS_DIR, config["filename"])
    if os.path.exists(save_path):
        size = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\n  模型已存在: {config['filename']} ({size:.1f} MB)")
        return True

    for url in config["urls"]:
        if download_file(url, save_path, config["name"]):
            return True
        print("  尝试下一个下载源...")

    print("\n  下载失败，请手动下载:")
    print(f"  访问: {config['urls'][0]}")
    print(f"  保存到: {save_path}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="猫咪姿态识别模型下载工具")
    parser.add_argument(
        "--model", "-m", type=str, default="rtmpose",
        choices=list(MODEL_CONFIGS.keys()),
        help="要下载的模型 (默认: rtmpose)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="列出已有模型文件")
    parser.add_argument("--all", "-a", action="store_true", help="下载所有模型")
    args = parser.parse_args()

    print("=" * 60)
    print("  猫咪姿态识别模型下载工具")
    print("=" * 60)

    if args.list:
        list_models()
        return 0

    if args.all:
        success = True
        for key in MODEL_CONFIGS:
            success = download_model(key) and success
        list_models()
        return 0 if success else 1

    config = MODEL_CONFIGS[args.model]
    print(f"\n  模型: {config['name']}")
    print(f"  说明: {config['description']}")
    print(f"  大小: ~{config['size_mb']} MB")
    if config.get("recommended"):
        print("  推荐: 是")

    success = download_model(args.model)
    list_models()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())