#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gradio as gr

# Make sure wandb never tries to init inside Spaces
os.environ["WANDB_MODE"] = "disabled"

# Optional: allow overriding MeCab binary path via Space secret/variable `MECAB_BIN`
MECAB_BIN = os.getenv("MECAB_BIN", "mecab")
os.environ["MECAB_BIN"] = MECAB_BIN


# Lazy-load model once per process
_model = None
_exp_info = None


def _ensure_model():
    global _model, _exp_info
    if _model is None:
        from infer import load_model

        result = load_model()
        if result is None:
            raise RuntimeError(
                "Model could not be loaded. Ensure sample_model/ exists with config.yaml and model.pt."
            )
        _model, _exp_info = result


def analyze(text: str):
    if not text or not text.strip():
        return "", []

    _ensure_model()

    from infer import predict_morphemes_from_text

    try:
        result = predict_morphemes_from_text(text.strip(), _model, _exp_info, silent=True)
        if not result:
            return "推論に失敗しました。", []
        results, optimal_morphemes = result

        # Prefer optimal morphemes (Viterbi decoded); fallback to raw results
        items = optimal_morphemes if optimal_morphemes else results
        tokens = [m.get("surface", "") for m in items]

        # Show a simple segmented string and a lightweight table
        segmented = " ".join(tokens)
        table = [
            {
                "surface": m.get("surface", ""),
                "pos": m.get("pos", "*"),
                "start": m.get("start_pos", ""),
                "end": m.get("end_pos", ""),
                "prob": round(float(m.get("probability", 0.0)), 3) if m.get("probability") is not None else "",
            }
            for m in items
        ]

        return segmented, table
    except Exception as e:
        return f"エラー: {e}", []


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Mecari Morpheme Analyzer

    日本語形態素解析（GATv2ベース）のデモ。テキストを入力すると、推定された形態素境界で分割して表示します。
    """
    )

    with gr.Row():
        inp = gr.Textbox(label="テキスト入力", placeholder="今日は良い天気ですね。", lines=3)
    with gr.Row():
        out_text = gr.Textbox(label="分割結果 (空白区切り)")
    out_tbl = gr.Dataframe(
        headers=["surface", "pos", "start", "end", "prob"],
        label="詳細",
        datatype=["str", "str", "number", "number", "number"],
        wrap=True,
    )

    btn = gr.Button("解析する")
    btn.click(fn=analyze, inputs=inp, outputs=[out_text, out_tbl])

    # Run a quick warm-up on Space start (optional and safe if MeCab is present)
    def _warmup():
        try:
            _ensure_model()
        except Exception:
            # Defer failure to first real request so Space still builds
            pass

    _warmup()

if __name__ == "__main__":
    demo.launch()

