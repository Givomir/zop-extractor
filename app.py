"""
HuggingFace Space frontend за ЗОП Екстрактор.

Качваш документи (PDF/DOCX) → екстракторът ги анализира с LLM → връща
структуриран JSON с ключовата информация за обществената поръчка.

Backend се избира чрез променливата LLM_BACKEND:
  - "hf"     → HuggingFace Inference Providers (за Space-а; изисква HF_TOKEN)
  - "ollama" → локален Ollama сървър (за разработка)
"""

import os
import json
import shutil
import tempfile
import logging
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

# На HuggingFace по подразбиране ползваме HF backend, освен ако не е зададено друго.
os.environ.setdefault("LLM_BACKEND", "hf")

from extractor import process_documents, ExtractorError, LLM_BACKEND, HF_MODEL

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}

# Подредба и етикети на полетата за показване
FIELD_LABELS = [
    ("възложител", "Възложител"),
    ("предмет", "Предмет"),
    ("прогнозна_стойност", "Прогнозна стойност"),
    ("начин_на_възлагане", "Начин на възлагане"),
    ("кратко_описание", "Кратко описание"),
    ("критерии_за_подбор", "Критерии за подбор"),
    ("изисквания_за_сертификация", "Изисквания за сертификация"),
    ("критерии_за_възлагане", "Критерии за възлагане"),
    ("краен_срок_за_подаване", "Краен срок за подаване"),
    ("дата_на_публикуване", "Дата на публикуване"),
]


def _format_value(val) -> str:
    """Форматира стойност за Markdown показване."""
    if val is None:
        return "_не е намерено в документите_"
    if isinstance(val, list):
        if not val:
            return "_не е намерено в документите_"
        return "\n".join(f"- {item}" for item in val)
    return str(val)


def _results_to_markdown(result: dict) -> str:
    """Превръща резултата в четим Markdown."""
    lines = ["## Извлечена информация\n"]

    # Линк към поръчката (най-отгоре, за референция)
    link = result.get("линк")
    if link:
        lines.append(f"🔗 **Линк към поръчката:** [{link}]({link})")
    else:
        lines.append("🔗 **Линк към поръчката:** _не е предоставен_")
    lines.append("")

    for key, label in FIELD_LABELS:
        val = result.get(key)
        lines.append(f"### {label}")
        lines.append(_format_value(val))
        lines.append("")

    # Допълнителна информация
    processed = result.get("обработени_файлове", [])
    skipped = result.get("пропуснати_файлове", [])
    if processed:
        lines.append("---")
        lines.append(f"**Обработени файлове:** {', '.join(processed)}")
    if skipped:
        lines.append(f"**Пропуснати файлове:** {', '.join(skipped)}")
    return "\n".join(lines)


def _cleanup_uploaded(files):
    """
    Трие файловете, които Gradio е качил в своя кеш (обикновено /tmp/gradio/...).
    Извиква се след обработка, за да не се трупат на диска на Space-а.
    """
    if not files:
        return
    for f in files:
        try:
            path = f.name if hasattr(f, "name") else f
            p = Path(path)
            if p.exists() and p.is_file():
                p.unlink()
                # ако родителската папка е празна Gradio temp папка — махаме я
                parent = p.parent
                if parent.exists() and "gradio" in str(parent).lower() and not any(parent.iterdir()):
                    parent.rmdir()
        except Exception as e:
            logger.warning(f"Не можах да изтрия качен файл: {e}")


def _cleanup_old_temp(max_age_seconds: int = 3600):
    """
    Защитен механизъм: трие наши временни папки (zop_*) и стари файлове в
    Gradio кеша, по-стари от max_age_seconds. Така дори при срив по средата
    на обработка, дискът не се препълва с времето.
    """
    import time
    now = time.time()
    roots = [Path(tempfile.gettempdir()), Path(tempfile.gettempdir()) / "gradio"]
    for root in roots:
        if not root.exists():
            continue
        for item in root.iterdir():
            try:
                # чистим само нашите temp папки и gradio кеша
                is_ours = item.name.startswith("zop_")
                is_gradio = root.name == "gradio"
                if not (is_ours or is_gradio):
                    continue
                age = now - item.stat().st_mtime
                if age < max_age_seconds:
                    continue
                if item.is_dir():
                    shutil.rmtree(str(item), ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception:
                pass


def extract(files, url):
    """
    Главната функция, която Gradio извиква. Приема качени файлове и
    незадължителен URL, връща (Markdown резюме, суров JSON).
    """
    if not files:
        return "⚠️ Моля качете поне един документ (PDF или DOCX).", "{}"

    # Защитно чистене на стари файлове преди да започнем (евтино)
    _cleanup_old_temp()

    # Копираме качените файлове във временна папка с истинските им имена
    tmp_dir = tempfile.mkdtemp(prefix="zop_")
    saved_paths = []
    skipped = []
    try:
        for f in files:
            src = Path(f.name if hasattr(f, "name") else f)
            ext = src.suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                dst = Path(tmp_dir) / src.name
                shutil.copy(str(src), str(dst))
                saved_paths.append(str(dst))
            else:
                skipped.append(src.name)

        if not saved_paths:
            return "⚠️ Нито един файл не е в поддържан формат (PDF, DOCX).", "{}"

        url = (url or "").strip() or None
        result = process_documents(saved_paths, url=url)
        result["обработени_файлове"] = [Path(p).name for p in saved_paths]
        if skipped:
            result["пропуснати_файлове"] = skipped

        markdown = _results_to_markdown(result)
        raw_json = json.dumps(result, ensure_ascii=False, indent=2)
        return markdown, raw_json

    except ExtractorError as e:
        return f"❌ {e}", "{}"
    except Exception as e:
        return f"❌ Неочаквана грешка: {e}", "{}"
    finally:
        # Трием и нашата temp папка, и качените от Gradio файлове
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _cleanup_uploaded(files)


# ── Интерфейс ────────────────────────────────────────────────────────────────

DESCRIPTION = f"""
# 📋 ЗОП Екстрактор

Автоматично извличане на ключова информация от документи за обществени поръчки
по ЗОП. Качи документите на поръчката (обявление, решение, договор, техническа
спецификация и др.) и системата ще извлече възложител, предмет, стойност,
критерии и срокове.

*Модел: `{HF_MODEL if LLM_BACKEND == "hf" else "локален Ollama"}`. Това е PoC —
резултатите може да съдържат грешки; винаги проверявай спрямо оригиналните документи.*
"""

with gr.Blocks(title="ЗОП Екстрактор") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            files_input = gr.File(
                label="Документи на поръчката (PDF / DOCX)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".doc"],
            )
            url_input = gr.Textbox(
                label="Линк към поръчката (незадължително)",
                placeholder="https://app.eop.bg/today/585570",
            )
            submit_btn = gr.Button("🔍 Извлечи информацията", variant="primary")

        with gr.Column(scale=2):
            output_md = gr.Markdown(label="Резултат")
            with gr.Accordion("Суров JSON", open=False):
                output_json = gr.Code(language="json", label="JSON")

    submit_btn.click(
        fn=extract,
        inputs=[files_input, url_input],
        outputs=[output_md, output_json],
    )

    gr.Markdown(
        "💡 *Съвет: качи няколко документа от една поръчка наведнъж — "
        "системата сравнява информацията между тях за по-точен резултат.*"
    )

if __name__ == "__main__":
    demo.launch()