#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import gradio as gr

# Ensure wandb never starts in Spaces
os.environ["WANDB_MODE"] = "disabled"

# Resolve MeCab binary for this process
_default_mecab = "/usr/bin/mecab" if os.path.exists("/usr/bin/mecab") else "mecab"
MECAB_BIN = os.getenv("MECAB_BIN", _default_mecab)
os.environ["MECAB_BIN"] = MECAB_BIN

# Lazy-loaded model
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


def _to_mecab_lines(results, optimal_morphemes=None) -> str:
    # Build MeCab-like output lines
    def mecab_features(m):
        pos = m.get("pos", "*")
        pos1 = m.get("pos_detail1", "*")
        pos2 = m.get("pos_detail2", "*")
        ctype = m.get("inflection_type", "*")
        cform = m.get("inflection_form", "*")
        base = m.get("base_form", m.get("lemma", "*")) or "*"
        # Mecari output includes reading as 7th field
        reading = m.get("reading", "*") or "*"
        return f"{pos},{pos1},{pos2},{ctype},{cform},{base},{reading}"

    items = (
        optimal_morphemes
        if optimal_morphemes
        else [
            {
                "surface": r.get("surface", ""),
                "pos": r.get("pos", "*"),
                "pos_detail1": "*",
                "pos_detail2": "*",
                "inflection_type": "*",
                "inflection_form": "*",
                "base_form": r.get("surface", ""),
                "reading": r.get("reading", "*"),
            }
            for r in results
        ]
    )

    lines = [f"{m.get('surface','')}\t{mecab_features(m)}" for m in items]
    lines.append("EOS")
    return "\n".join(lines)


def mecab_plain(text: str) -> str:
    """Run system MeCab and return its raw parsing (surface\tCSV ...\nEOS)."""
    try:
        from mecari.analyzers.mecab import MeCabAnalyzer

        analyzer = MeCabAnalyzer()
        mecab_bin = os.getenv("MECAB_BIN", analyzer.mecab_bin)
        args = [mecab_bin]
        if isinstance(analyzer.jumandic_path, str) and os.path.isdir(analyzer.jumandic_path):
            args += ["-d", analyzer.jumandic_path]
        p = subprocess.run(args, input=text, text=True, capture_output=True)
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        if p.returncode != 0:
            return out.strip() or f"mecab error rc={p.returncode}"
        # Trim extra tail fields (e.g., カテゴリ:*, ドメイン:*) and keep first 6 features
        lines = []
        for line in out.splitlines():
            if not line or line.strip() == "EOS":
                lines.append("EOS")
                continue
            if "\t" in line:
                surface, feats = line.split("\t", 1)
                parts = [s.strip() for s in feats.split(",")]
                trimmed = parts[:6]
                while len(trimmed) < 6:
                    trimmed.append("*")
                lines.append(f"{surface}\t{','.join(trimmed)}")
            else:
                lines.append(line)
        # Ensure trailing EOS only once
        if not lines or lines[-1] != "EOS":
            lines.append("EOS")
        return "\n".join(lines)
    except FileNotFoundError:
        return "MeCabバイナリが見つかりません（MECAB_BINやpackages.txtを確認）。"
    except Exception as e:
        return f"mecab実行時エラー: {e}"


def analyze(text: str):
    if not text or not text.strip():
        return "", ""

    try:
        _ensure_model()
        from infer import predict_morphemes_from_text

        text = text.strip()
        result = predict_morphemes_from_text(text, _model, _exp_info, silent=True)
        if not result:
            return "推論に失敗しました。", mecab_plain(text)
        results, optimal_morphemes = result
        mecari_out = _to_mecab_lines(results, optimal_morphemes)
        mecab_out = mecab_plain(text)
        return mecari_out, mecab_out
    except FileNotFoundError:
        return (
            "MeCabが見つかりません。Spaceのpackages.txtに 'mecab' と 'mecab-jumandic-utf8' を含めてビルドし直すか、\n"
            "変数 MECAB_BIN=/usr/bin/mecab を設定してください。"
        ), ""
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        return f"エラー: {e}\n\n{tb}", ""


FONT_CSS = """
/* Prefer common system fonts for Latin text */
body, .gradio-container, .prose, textarea, input, button,
.gr-text-input input, .gr-text-input textarea, .gr-textbox textarea {
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans',
               'Helvetica Neue', Arial, 'Apple Color Emoji', 'Segoe UI Emoji',
               sans-serif !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=FONT_CSS) as demo:
    gr.Markdown(
        """
    # Mecari Morpheme Analyzer

    GNNベースの形態素解析器"Mecari"のデモです。github: https://github.com/zbller/Mecari
    """
    )

    with gr.Row():
        inp = gr.Textbox(label="テキスト入力", value="とうきょうに行った", placeholder="とうきょうに行った", lines=3)
    btn = gr.Button("解析する")
    with gr.Row():
        out_mecari = gr.Textbox(label="Mecari", lines=10)
        out_mecab = gr.Textbox(label="MeCab（Jumandic）", lines=10)
    btn.click(fn=analyze, inputs=inp, outputs=[out_mecari, out_mecab])

    # Optional warm-up
    def _warmup():
        try:
            _ensure_model()
        except Exception:
            pass

    _warmup()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
