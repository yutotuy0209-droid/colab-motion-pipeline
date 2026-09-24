"""Hybrid AI Experiment Advisor (Local Ollama & Colab AI API).

Connects to either:
1. Local Ollama instance (sotsuron-assistant / qwen2.5:7b)
2. Remote Colab FastAPI instance (OpenAI-compatible /v1/chat/completions)

Automatically aggregates experiment parameters (backlash, runout, G-code rest windows)
and queries the target AI model for engineering insights and advisor notes.
"""

# ==============================================================================
# [FOR CODEX REFACTOR]: Hybrid AI Client for Local Ollama & Colab LLM
# [INPUT/OUTPUT SPEC]:
#   - Input:
#       source: str ('ollama' or 'colab')
#       experiment_data: dict containing backlash, scale, runout, and G-code
#       custom_prompt: Optional[str]
#   - Output:
#       advice: str, formatted engineering review for Prof. Li
# [ALGORITHM INTENT]:
#   - Bridge local offline LLM (Ollama) with cloud GPU LLM (Colab) under unified API.
#   - Ensure high repeatability and zero vendor lock-in with standard HTTP JSON protocols.
# [TODO FOR CODEX]:
#   - Add async streaming output support (Server-Sent Events) for real-time terminal display.
#   - Integrate token usage and latency telemetry into JSON loggers.
# ==============================================================================

import argparse
import json
import sys
import urllib.request
import urllib.error

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_EXPERIMENT_SUMMARY = {
    "backlash_um": 0.865,
    "backlash_target_um": 1.000,
    "forward_scale_alpha": 1.0015,
    "backward_scale_alpha": 1.0015,
    "repeatability_3sigma_um": 0.118,
    "rotation_runout_pv_um": 0.957,
    "gcode_rest_windows": "1.5s - 3.0s (at X=10um), 4.5s - 6.0s (at X=20um)",
    "poc_subpixel_err_px": 0.050
}


def build_engineering_prompt(exp_data: dict, extra_question: str = "") -> str:
    prompt = f"""あなたは精密位置決め機械・画像計測工学の専門AIアシスタントです。
以下の実機実験同定データを評価し、指導教員（李先生）への報告に向けた工学的考察と具体的な改善アドバイスを論理的かつ簡潔に述べてください。

【最新の実機実験同定データ】
- 同定反転バックラッシ量: {exp_data.get('backlash_um', 0.865):.3f} µm (許容目安: {exp_data.get('backlash_target_um', 1.0):.1f} µm以下)
- 往路スケール校正傾き: {exp_data.get('forward_scale_alpha', 1.0015):.4f} (設計指令比)
- 復路スケール校正傾き: {exp_data.get('backward_scale_alpha', 1.0015):.4f}
- 3-Sigma 再現性ばらつき: {exp_data.get('repeatability_3sigma_um', 0.118):.3f} µm
- テーブル回転中心偏心振れ回り (P-V): {exp_data.get('rotation_runout_pv_um', 0.957):.3f} µm
- Gコード安定撮影静止窓: {exp_data.get('gcode_rest_windows', '1.5s - 3.0s')}
- 2D-POC サブピクセル推定誤差: {exp_data.get('poc_subpixel_err_px', 0.050):.3f} px
"""
    if extra_question:
        prompt += f"\n【追加の検討要請・質問】\n{extra_question}\n"
    return prompt


def query_ollama(prompt: str, model: str = "sotsuron-assistant:latest", host: str = "http://localhost:11434") -> str:
    """Query local Ollama server."""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json.get("response", "")
    except urllib.error.URLError as e:
        return f"[ERROR] Ollama connection failed ({e}). Is 'ollama serve' running on {host}?"


def query_colab(prompt: str, colab_url: str = "http://localhost:8000") -> str:
    """Query Colab OpenAI-compatible API endpoint."""
    url = f"{colab_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "qwen2.5-0.5b",
        "messages": [
            {"role": "system", "content": "あなたは精密機械工学の専門AIです。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"[ERROR] Colab API connection failed ({e}). Check if the Colab FastAPI server is running."


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid AI Experiment Advisor")
    parser.add_argument("--source", choices=["ollama", "colab"], default="ollama", help="Target AI engine")
    parser.add_argument("--model", type=str, default="sotsuron-assistant:latest", help="Ollama model name")
    parser.add_argument("--colab-url", type=str, default="http://localhost:8000", help="Colab endpoint URL")
    parser.add_argument("--question", type=str, default="", help="Optional extra question for the advisor")
    args = parser.parse_args()

    print("=" * 68)
    print(f"  [AI Advisor] Engine: {args.source.upper()} | Model: {args.model}")
    print("  Aggregating experiment metrics...")
    print("=" * 68)

    prompt = build_engineering_prompt(DEFAULT_EXPERIMENT_SUMMARY, args.question)

    if args.source == "ollama":
        print(f"Connecting to Local Ollama ({args.model})... Please wait...")
        reply = query_ollama(prompt, model=args.model)
    else:
        print(f"Connecting to Colab AI API ({args.colab_url})... Please wait...")
        reply = query_colab(prompt, colab_url=args.colab_url)

    print("\n" + "=" * 68)
    print("  【AI 実験考察 ＆ 李先生向けアドバイス】")
    print("=" * 68)
    print(reply.strip())
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
