"""
ONNX导出精度校验模块
用于在导出过程中对比PyTorch模型和ONNX模型的输出，定位精度损失位置
"""
import os
import torch
import numpy as np
import onnxruntime
from typing import Dict, Tuple, Optional, Any
import json


class ONNXValidator:
    """ONNX导出精度校验器"""

    def __init__(self, output_dir: str, onnx_device: str = "cpu"):
        """
        初始化校验器

        Args:
            output_dir: ONNX输出目录
            onnx_device: ONNX推理设备 ("cpu" 或 "cuda")
        """
        self.output_dir = output_dir
        self.onnx_device = onnx_device
        self.validation_results = []

        # 设置ONNX运行时选项
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        if onnx_device == "cuda":
            self.providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

    def validate_model(
        self,
        model_name: str,
        onnx_path: str,
        pytorch_model: torch.nn.Module,
        dummy_inputs: Dict[str, torch.Tensor],
        output_names: list,
        rtol: float = 1e-3,
        atol: float = 1e-5
    ) -> bool:
        """
        校验单个模型的输出精度

        Args:
            model_name: 模型名称
            onnx_path: ONNX模型路径
            pytorch_model: PyTorch模型
            dummy_inputs: 虚拟输入字典 {input_name: tensor}
            output_names: 输出名称列表
            rtol: 相对误差阈值
            atol: 绝对误差阈值

        Returns:
            是否通过校验
        """
        print(f"\n{'='*60}")
        print(f"校验模型: {model_name}")
        print(f"{'='*60}")

        if not os.path.exists(onnx_path):
            print(f"❌ ONNX模型不存在: {onnx_path}")
            return False

        # 确保模型在评估模式
        pytorch_model.eval()

        # 准备PyTorch输入
        pt_inputs = []
        input_names = list(dummy_inputs.keys())
        for name in input_names:
            tensor = dummy_inputs[name]
            # 如果是GPU上的模型，确保输入在GPU上
            if hasattr(pytorch_model, 'parameters'):
                try:
                    next(pytorch_model.parameters())
                    device = next(pytorch_model.parameters()).device
                    if tensor.device != device:
                        tensor = tensor.to(device)
                except StopIteration:
                    pass
            pt_inputs.append(tensor)

        # PyTorch前向传播
        with torch.no_grad():
            try:
                if len(pt_inputs) == 1:
                    pt_outputs = pytorch_model(pt_inputs[0])
                else:
                    pt_outputs = pytorch_model(*pt_inputs)

                # 如果输出是元组，转换为列表
                if isinstance(pt_outputs, tuple):
                    pt_outputs = list(pt_outputs)
                elif not isinstance(pt_outputs, list):
                    pt_outputs = [pt_outputs]
            except Exception as e:
                print(f"❌ PyTorch前向传播失败: {e}")
                return False

        # 加载ONNX模型并推理
        try:
            ort_session = onnxruntime.InferenceSession(
                onnx_path,
                sess_options=onnxruntime.SessionOptions(),
                providers=self.providers
            )
        except Exception as e:
            print(f"❌ 加载ONNX模型失败: {e}")
            return False

        # 准备ONNX输入
        ort_inputs = {}
        for name, tensor in dummy_inputs.items():
            # 转换为numpy
            arr = tensor.detach().cpu().numpy()
            ort_inputs[name] = arr

        try:
            ort_outputs = ort_session.run(output_names, ort_inputs)
        except Exception as e:
            print(f"❌ ONNX前向传播失败: {e}")
            return False

        # 对比输出
        all_passed = True
        for i, (pt_output, ort_output, out_name) in enumerate(zip(pt_outputs, ort_outputs, output_names)):
            passed, metrics = self._compare_tensors(
                pt_output, ort_output, f"{out_name}_{i}", rtol, atol
            )
            if not passed:
                all_passed = False

            # 保存结果
            result = {
                "model_name": model_name,
                "output_name": out_name,
                "passed": passed,
                "metrics": metrics
            }
            self.validation_results.append(result)

        status = "✅ 通过" if all_passed else "❌ 失败"
        print(f"\n{status} {model_name} 校验完成")
        print(f"{'='*60}\n")

        return all_passed

    def _compare_tensors(
        self,
        pt_tensor: torch.Tensor,
        ort_tensor: np.ndarray,
        output_name: str,
        rtol: float,
        atol: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        对比两个张量的精度

        Args:
            pt_tensor: PyTorch张量
            ort_tensor: ONNX输出numpy数组
            output_name: 输出名称
            rtol: 相对误差阈值
            atol: 绝对误差阈值

        Returns:
            (是否通过, 指标字典)
        """
        # 转换为numpy
        pt_np = pt_tensor.detach().cpu().numpy()
        ort_np = ort_tensor

        # 确保形状一致
        if pt_np.shape != ort_np.shape:
            print(f"  ⚠️  输出形状不匹配: PyTorch {pt_np.shape} vs ONNX {ort_np.shape}")
            return False, {"error": "shape_mismatch"}

        # 计算各种误差指标
        abs_diff = np.abs(pt_np - ort_np)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        # 相对误差（避免除零）
        denominator = np.maximum(np.abs(pt_np), np.abs(ort_np))
        rel_diff = abs_diff / np.maximum(denominator, 1e-10)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # MSE
        mse = np.mean((pt_np - ort_np) ** 2)
        rmse = np.sqrt(mse)

        # 余弦相似度（用于评估特征嵌入质量）
        if pt_np.ndim >= 2:
            pt_flat = pt_np.reshape(pt_np.shape[0], -1)
            ort_flat = ort_np.reshape(ort_np.shape[0], -1)

            # 归一化
            pt_norm = pt_flat / (np.linalg.norm(pt_flat, axis=1, keepdims=True) + 1e-10)
            ort_norm = ort_flat / (np.linalg.norm(ort_flat, axis=1, keepdims=True) + 1e-10)

            # 批量余弦相似度
            cosine_sim = np.mean(np.sum(pt_norm * ort_norm, axis=1))
        else:
            cosine_sim = np.corrcoef(pt_np.flatten(), ort_np.flatten())[0, 1]

        metrics = {
            "max_abs_diff": float(max_abs_diff),
            "mean_abs_diff": float(mean_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "mean_rel_diff": float(mean_rel_diff),
            "mse": float(mse),
            "rmse": float(rmse),
            "cosine_similarity": float(cosine_sim) if not np.isnan(cosine_sim) else 0.0,
            "pt_range": [float(np.min(pt_np)), float(np.max(pt_np))],
            "ort_range": [float(np.min(ort_np)), float(np.max(ort_np))],
            "pt_mean": float(np.mean(pt_np)),
            "ort_mean": float(np.mean(ort_np)),
        }

        # 打印详细报告
        print(f"\n  输出: {output_name}")
        print(f"  形状: {pt_np.shape}")
        print(f"  PyTorch 范围: [{metrics['pt_range'][0]:.6f}, {metrics['pt_range'][1]:.6f}], 均值: {metrics['pt_mean']:.6f}")
        print(f"  ONNX    范围: [{metrics['ort_range'][0]:.6f}, {metrics['ort_range'][1]:.6f}], 均值: {metrics['ort_mean']:.6f}")
        print(f"  最大绝对误差: {max_abs_diff:.6e}")
        print(f"  平均绝对误差: {mean_abs_diff:.6e}")
        print(f"  最大相对误差: {max_rel_diff:.6%}")
        print(f"  平均相对误差: {mean_rel_diff:.6%}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  余弦相似度: {cosine_sim:.6f}")

        # 判断是否通过
        passed = (max_rel_diff < rtol) and (max_abs_diff < atol)

        # 对于声纹嵌入，余弦相似度更重要
        if "sv" in output_name.lower() or "embedding" in output_name.lower():
            if cosine_sim < 0.99:
                print(f"  ⚠️  声纹嵌入相似度较低，可能导致音色失真！")
                passed = False

        return passed, metrics

    def print_summary(self):
        """打印校验摘要"""
        print(f"\n{'='*80}")
        print("ONNX导出精度校验摘要")
        print(f"{'='*80}\n")

        failed_count = 0
        total_count = len(self.validation_results)

        for result in self.validation_results:
            model_name = result["model_name"]
            output_name = result["output_name"]
            passed = result["passed"]
            metrics = result["metrics"]

            status = "✅" if passed else "❌"
            print(f"{status} {model_name} - {output_name}")

            if not passed:
                failed_count += 1
                print(f"   最大相对误差: {metrics['max_rel_diff']:.6%}")
                print(f"   余弦相似度: {metrics['cosine_similarity']:.6f}")
                if "cosine_similarity" in metrics and metrics['cosine_similarity'] < 0.95:
                    print(f"   ⚠️  警告: 相似度过低，可能导致严重失真！")

        print(f"\n总计: {total_count} 个输出, {failed_count} 个失败, {total_count - failed_count} 个通过")

        if failed_count > 0:
            print(f"\n❌ 存在精度损失，建议检查失败的模块")
            # 找出损失最大的模块
            print("\n🔍 损失最大的模块:")
            sorted_results = sorted(
                self.validation_results,
                key=lambda x: x["metrics"].get("cosine_similarity", 1.0)
            )
            for result in sorted_results[:3]:  # 显示最差的3个
                metrics = result["metrics"]
                print(f"  - {result['model_name']}/{result['output_name']}: "
                      f"余弦相似度={metrics['cosine_similarity']:.6f}, "
                      f"最大相对误差={metrics['max_rel_diff']:.6%}")
        else:
            print(f"\n✅ 所有模块校验通过！")

        print(f"{'='*80}\n")

    def save_report(self, output_path: str = None):
        """保存详细报告到JSON文件"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "validation_report.json")

        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(v) for v in obj]
            return obj

        # 转换validation_results中的numpy类型
        converted_results = convert_numpy_types(self.validation_results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)

        print(f"详细报告已保存到: {output_path}")


def create_validation_audio(
    ref_wav_path: str,
    target_sr: int = 16000,
    duration: float = 3.0
) -> np.ndarray:
    """
    创建用于校验的音频数据

    Args:
        ref_wav_path: 参考音频路径
        target_sr: 目标采样率
        duration: 目标时长（秒）

    Returns:
        音频numpy数组
    """
    import librosa

    # 加载音频
    audio, sr = librosa.load(ref_wav_path, sr=target_sr)

    # 截取或填充到目标时长
    target_length = int(target_sr * duration)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    return audio