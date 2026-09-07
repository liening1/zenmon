"""zenmon.ai_inspector: 本地大模型与深度学习工作流智能感知引擎.

自动识别当前系统中运行的 AI / ML / LLM 进程 (Ollama, vLLM, PyTorch, ComfyUI 等),
提取显存与算力画像, 专为现代 AI 开发者与算力工作站打造.
"""
import re

# AI / ML 签名库与对应的语义标签
AI_SIGNATURES = [
    (r"\bollama(_llama_server)?\b", "Ollama"),
    (r"\bvllm(\.entrypoints)?\b", "vLLM"),
    (r"\bllama(-server|\.cpp)?\b", "llama.cpp"),
    (r"\bcomfyui\b", "ComfyUI"),
    (r"\b(stable-diffusion|sd-webui|webui\.py)\b", "SD-WebUI"),
    (r"\bdeepspeed\b", "DeepSpeed"),
    (r"\baccelerate\b", "Accelerate"),
    (r"\btriton\b", "Triton"),
    (r"\btext-generation-launcher\b", "TGI"),
    (r"\btorchrun\b", "Torchrun"),
    (r"\b(python\d*.*[-_](train|infer|eval|finetune|generate|serve)\.py)\b", "ML-Train/Infer"),
    (r"\b(torch|tensorflow|jax|tensorrt)\b", "PyTorch/TF"),
    (r"\bjupyter-(lab|notebook)\b", "Jupyter"),
    (r"\bwhisper\b", "Whisper"),
]

COMPILED_SIGS = [(re.compile(pat, re.IGNORECASE), label) for pat, label in AI_SIGNATURES]

def detect_ai_tag(cmdline: str, comm: str = "") -> str:
    """检测进程命令行或名称是否匹配已知 AI 工作流, 返回标签名称 (如 'Ollama') 或 None."""
    full_text = f"{comm} {cmdline}"
    for regex, label in COMPILED_SIGS:
        if regex.search(full_text):
            return label
    return None

def analyze_ai_summary(procs: list, gpu_procs: list = None) -> dict:
    """聚合分析系统全局 AI 算力与显存消耗画像."""
    ai_procs = []
    total_ai_cpu = 0.0
    total_ai_mem_gb = 0.0
    detected_frameworks = set()

    for p in procs:
        tag = p.get("ai_tag")
        if tag:
            ai_procs.append(p)
            total_ai_cpu += p.get("cpu", 0.0)
            total_ai_mem_gb += (p.get("res", 0) / (1024 * 1024))
            detected_frameworks.add(tag)

    ai_vram_mb = 0.0
    if gpu_procs:
        for gp in gpu_procs:
            if gp.get("ai_tag"):
                ai_vram_mb += gp.get("mem_mb", 0.0)

    return {
        "count": len(ai_procs),
        "frameworks": sorted(list(detected_frameworks)),
        "total_cpu_pct": total_ai_cpu,
        "total_sys_mem_gb": total_ai_mem_gb,
        "total_vram_mb": ai_vram_mb,
    }
