"""下载 RTMPose-M AP-10K 动物姿态模型（ONNX）

RTMPose-M 在 AP-10K 数据集（54 种动物，含猫/狗）上训练，是本项目
默认的动物姿态模型。脚本从 HuggingFace 镜像下载 MMDeploy 导出的
ONNX SDK 压缩包，解压出 end2end.onnx 并保存为 models/rtmpose_ap10k.onnx。

用法:
    python download_rtmpose.py
"""

import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rtmpose_ap10k.onnx")
ARCHIVE_NAME = "rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip"
ARCHIVE_DIR = "rtmposev1/onnx_sdk"

# 优先使用国内可访问的 HF 镜像，失败后回退官方站点
MODEL_URLS = [
    f"https://hf-mirror.com/Tau-J/RTMPose/resolve/main/{ARCHIVE_DIR}/{ARCHIVE_NAME}?download=true",
    f"https://huggingface.co/Tau-J/RTMPose/resolve/main/{ARCHIVE_DIR}/{ARCHIVE_NAME}?download=true",
]

MIN_EXPECTED_BYTES = 40 * 1024 * 1024  # 压缩包约 48MB，ONNX 约 52MB
CHUNK_SIZE = 1024 * 256
RETRIES_PER_URL = 3


def _format_mb(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def _download(url: str) -> str:
    """下载压缩包到临时文件，返回临时文件路径。"""
    for attempt in range(1, RETRIES_PER_URL + 1):
        try:
            print(f"\n[INFO] 下载中（第 {attempt} 次尝试）: {url}")
            request = urllib.request.Request(
                url, headers={"User-Agent": "PetBehaviorAnalyzer/2.1"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("content-length", 0) or 0)
                if total:
                    print(f"[INFO] 文件大小: {_format_mb(total)}")

                fd, temp_path = tempfile.mkstemp(suffix=".zip", prefix="rtmpose_")
                downloaded = 0
                last_percent = -5
                with os.fdopen(fd, "wb") as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded * 100 // total
                            if percent - last_percent >= 5:
                                last_percent = percent
                                sys.stdout.write(
                                    f"\r[INFO] 进度: {percent}% "
                                    f"({_format_mb(downloaded)} / {_format_mb(total)})"
                                )
                                sys.stdout.flush()

            print(f"\n[OK] 下载完成: {_format_mb(os.path.getsize(temp_path))}")
            if os.path.getsize(temp_path) < MIN_EXPECTED_BYTES:
                raise ValueError("下载文件过小，可能不是有效的模型压缩包")
            return temp_path
        except Exception as error:
            print(f"\n[WARN] 下载失败: {error}")
            if attempt < RETRIES_PER_URL:
                time.sleep(2 * attempt)

    raise RuntimeError(f"下载失败: {url}")


def _extract_onnx(archive_path: str, target_path: str) -> None:
    """从 SDK 压缩包中解压出 end2end.onnx。"""
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("end2end.onnx")]
        if not candidates:
            raise RuntimeError("压缩包中未找到 end2end.onnx")

        # 优先取 rtmpose_onnx 目录下的模型文件
        member = next(
            (name for name in candidates if "rtmpose_onnx" in name),
            candidates[0],
        )
        print(f"[INFO] 解压模型: {member}")
        with archive.open(member) as source, open(target_path, "wb") as destination:
            shutil.copyfileobj(source, destination)


def _validate_model(model_path: str) -> bool:
    """用 onnxruntime 校验模型输入/输出形状。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] 未安装 onnxruntime，跳过模型校验")
        return True

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    outputs = session.get_outputs()
    if len(input_info.shape) != 4 or len(outputs) < 2:
        raise RuntimeError(f"模型结构异常: 输入 {input_info.shape}, 输出 {len(outputs)} 个")
    output_desc = ", ".join(f"{item.name}{item.shape}" for item in outputs)
    print(f"[OK] 模型校验通过: 输入 {input_info.name}{input_info.shape}，输出 {output_desc}")
    return True


def main() -> int:
    print("=" * 66)
    print("RTMPose-M AP-10K 动物姿态模型（ONNX）下载工具")
    print("=" * 66)

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_EXPECTED_BYTES:
        print(f"\n[OK] 模型已存在: {MODEL_PATH}")
        print(f"     文件大小: {_format_mb(os.path.getsize(MODEL_PATH))}")
        _validate_model(MODEL_PATH)
        print("\n如需重新下载，请删除该文件后重新运行此脚本。")
        return 0

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for url in MODEL_URLS:
        archive_path = None
        try:
            archive_path = _download(url)
            _extract_onnx(archive_path, MODEL_PATH)
            if os.path.getsize(MODEL_PATH) < MIN_EXPECTED_BYTES:
                raise RuntimeError("解压后的模型文件大小异常")
            _validate_model(MODEL_PATH)
            print(f"\n[OK] 模型已保存: {MODEL_PATH}")
            print(f"     文件大小: {_format_mb(os.path.getsize(MODEL_PATH))}")
            print("\n启动应用即可自动加载该模型。")
            return 0
        except Exception as error:
            print(f"\n[FAIL] 该下载源不可用: {error}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
        finally:
            if archive_path and os.path.exists(archive_path):
                os.remove(archive_path)

    print("\n" + "=" * 66)
    print("[FAIL] 所有下载源均失败，请手动下载：")
    print(f"1. 访问 https://hf-mirror.com/Tau-J/RTMPose 或 https://huggingface.co/Tau-J/RTMPose")
    print(f"2. 进入 {ARCHIVE_DIR} 目录，下载 {ARCHIVE_NAME}")
    print("3. 解压压缩包，将其中的 end2end.onnx 复制到:")
    print(f"   {MODEL_PATH}")
    print("=" * 66)
    return 1


if __name__ == "__main__":
    sys.exit(main())