"""
Извличане на информация от документи за обществени поръчки.
Стратегия: всеки документ се разделя на чънкове, всеки чънк се обработва
поотделно, резултатите се комбинират (по-дълга/по-пълна стойност печели).
"""

import os
import httpx
import json
import re
import logging
import pdfplumber
from docx import Document
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Избор на backend за LLM ──────────────────────────────────────────────────
# LLM_BACKEND="ollama" (локално, по подразбиране) или "hf" (HuggingFace router).
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()

# Ollama (локално)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "todorov/bggpt:Gemma-3-4B-IT-Q4_K_M")

# HuggingFace Inference Providers (router, OpenAI-съвместим)
HF_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Брой паралелни заявки към модела. 1 = последователно (най-безопасно за
# слаби машини). 2-3 ускорява значително, ако backend-ът издържа.
# За HF API внимавай с rate limits — стойност 2 е разумна.
MAX_PARALLEL = int(os.environ.get("LLM_PARALLEL", os.environ.get("OLLAMA_PARALLEL", "2")))

# Минимален размер на чънк — по-малки остатъци се сливат с предходния,
# за да не правим излишни заявки за 300 символа.
MIN_CHUNK_SIZE = 1500


CHUNK_SIZE = 6000


class ExtractorError(Exception):
    pass


def read_pdf(filepath: str) -> str:
    text = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n\n".join(text)


def read_docx(filepath: str) -> str:
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_document(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return read_pdf(filepath)
    elif ext in (".docx", ".doc"):
        return read_docx(filepath)
    else:
        raise ExtractorError(f"Неподдържан формат: {ext}")


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Разделя текста на чънкове до ~chunk_size знака, без да реже параграфи.
    Ако един параграф сам по себе си е по-дълъг от chunk_size, той се реже
    по изречения, а в краен случай — насилствено.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Параграфът се събира в текущия чънк
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
            continue

        # Текущият чънк е пълен — затваряме го
        if current:
            chunks.append(current)
            current = ""

        # Параграфът се събира сам в нов чънк
        if len(para) <= chunk_size:
            current = para
            continue

        # Параграфът е твърде дълъг — режем по изречения
        for piece in _split_oversized(para, chunk_size):
            if len(current) + len(piece) + 1 <= chunk_size:
                current = f"{current} {piece}" if current else piece
            else:
                if current:
                    chunks.append(current)
                current = piece

    if current:
        chunks.append(current)

    return _merge_small_chunks(chunks, chunk_size)


def _merge_small_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    """
    Слива чънкове по-малки от MIN_CHUNK_SIZE с предходния (ако се събират),
    за да не правим излишни LLM заявки за съвсем малки остатъци.
    """
    if len(chunks) <= 1:
        return chunks

    merged = []
    for c in chunks:
        if (merged
                and len(c) < MIN_CHUNK_SIZE
                and len(merged[-1]) + len(c) + 2 <= chunk_size):
            merged[-1] = merged[-1] + "\n\n" + c
        else:
            merged.append(c)
    return merged


def _split_oversized(para: str, chunk_size: int) -> list[str]:
    """Реже свръхдълъг параграф по изречения, после насилствено ако трябва."""
    sentences = re.split(r'(?<=[.!?])\s+', para)
    pieces = []
    for s in sentences:
        if len(s) <= chunk_size:
            pieces.append(s)
        else:
            # Насилствено рязане на парчета от chunk_size
            for i in range(0, len(s), chunk_size):
                pieces.append(s[i:i + chunk_size])
    return pieces


def fix_inner_quotes(raw: str) -> str:
    """
    Поправя вложени прави кавички вътре в стойностите на JSON стринг.
    Покрива два чести случая от BgGPT:
      1. "възложител": "Топлофикация София" ЕАД,
         → текст след затварящата кавичка се прибира ВЪТРЕ в стойността
      2. "предмет": "... „Топлофикация София" ЕАД /ЦУ/",
         → вложените прави кавички се escape-ват като \\"

    Работи както за многоредов, така и за едноредов JSON.
    """
    lines = raw.split("\n")
    if len(lines) > 1:
        return _fix_inner_quotes_multiline(raw)
    return _fix_inner_quotes_singleline(raw)


def _fix_inner_quotes_multiline(raw: str) -> str:
    """Ред по ред: всеки ред носи едно поле."""
    lines = raw.split("\n")
    fixed = []
    key_re = re.compile(r'^(\s*"[^"]+"\s*:\s*)(.*?)(\s*,?\s*)$')

    for line in lines:
        m = key_re.match(line)
        if not m:
            fixed.append(line)
            continue
        prefix, body, trailing = m.group(1), m.group(2), m.group(3)
        if not body.startswith('"'):
            fixed.append(line)
            continue
        last_q = body.rfind('"')
        if last_q == 0:
            fixed.append(line)
            continue
        inner = body[1:last_q]
        after = body[last_q + 1:]
        if after.strip():
            inner = inner + after
        inner = inner.replace('"', '\\"')
        fixed.append(f'{prefix}"{inner}"{trailing}')

    return "\n".join(fixed)


def _fix_inner_quotes_singleline(raw: str) -> str:
    """
    Едноредов JSON: разбиваме го на отделни редове по границите на полетата
    (всяко поле започва с  "ключ":  ) и после ползваме многоредовата логика.
    Това избягва нерешимата нееднозначност при символ-по-символ парсене.
    """
    # Слагаме нов ред преди всеки ключ:  , "ключ":  →  ,\n"ключ":
    # Ключ = кирилица/латиница/долна черта в кавички, следван от :
    spaced = re.sub(r',\s*("(?:[^"\\]|\\.)*?"\s*:)', r',\n\1', raw)
    # Отделяме и отварящата/затварящата скоба на собствени редове
    spaced = re.sub(r'^\s*\{', '{\n', spaced)
    spaced = re.sub(r'\}\s*$', '\n}', spaced)
    if spaced.count("\n") > 1:
        return _fix_inner_quotes_multiline(spaced)
    return raw  # не успяхме да сегментираме — връщаме оригинала


def repair_json(raw: str) -> str:
    raw = raw.replace('„', '"').replace('\u201e', '"')
    raw = raw.replace('\u201c', '"').replace('\u201d', '"')
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw


def repair_structural(raw: str) -> str:
    """
    По-агресивни структурни поправки за типични грешки от малки модели:
    - липсваща запетая между две полета (   "a": null⏎  "b": ...   )
    - отрязан/непълен JSON (затваряме висящ стринг и скоба)
    """
    s = raw.strip()

    # Махаме code-block огражденията, ако са останали
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)

    # Липсваща запетая: стойност, последвана от нов ред и нов ключ без запетая.
    # напр.  null⏎  "ключ":   или  "стойност"⏎  "ключ":
    s = re.sub(r'(null|true|false|"[^"\n]*"|\d)\s*\n(\s*")', r'\1,\n\2', s)

    # Ако JSON изглежда отрязан (не завършва с }), опитваме да го затворим.
    if not s.rstrip().endswith('}'):
        quote_count = len(re.findall(r'(?<!\\)"', s))
        if quote_count % 2 == 1:
            # висящ стринг — затваряме го
            s = s.rstrip().rstrip(',') + '"'
        s = s.rstrip().rstrip(',')
        open_braces = s.count('{') - s.count('}')
        s = s + ('}' * max(0, open_braces))

    return s


def extract_fields_individually(raw: str) -> dict:
    """
    Последна мярка: извличаме всяко известно поле поотделно с regex, така че
    едно счупено поле да не проваля останалите. Връща само намереното.
    """
    result = {}

    for field in FIELDS:
        # стрингова стойност:  "поле": "....."
        m = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if m:
            val = m.group(1).replace('\\"', '"').strip()
            if val:
                result[field] = val
            continue

        # списък:  "поле": [ ... ]
        m = re.search(rf'"{re.escape(field)}"\s*:\s*(\[[^\]]*\])', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(repair_json(m.group(1)))
                if parsed:
                    result[field] = parsed
            except (json.JSONDecodeError, ValueError):
                pass

    return result


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    attempts = [
        raw,
        repair_json(raw),
        fix_inner_quotes(raw),
        repair_json(fix_inner_quotes(raw)),
        repair_structural(raw),
        repair_json(repair_structural(raw)),
        fix_inner_quotes(repair_structural(raw)),
    ]
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Извличане на JSON от code-block или от текст, после пак всички поправки
    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"]:
        match = re.search(pattern, raw)
        if match:
            candidate = match.group(1).strip()
            for attempt in [
                candidate,
                repair_json(candidate),
                fix_inner_quotes(candidate),
                repair_json(fix_inner_quotes(candidate)),
                repair_structural(candidate),
                repair_json(repair_structural(candidate)),
            ]:
                try:
                    parsed = json.loads(attempt)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

    # Последна мярка: спасяваме каквото може поле по поле
    salvaged = extract_fields_individually(raw)
    if salvaged:
        logger.warning(f"  ! Частично спасени полета: {list(salvaged.keys())}")
        return salvaged

    logger.error(f"Не може да се парсне JSON: {raw[:200]}")
    return {}


SYSTEM_PROMPT = """Ти си асистент за анализ на документи за обществени поръчки по ЗОП.
Извличаш конкретна информация и я връщаш САМО като валиден JSON обект.
Не добавяй никакъв текст извън JSON. Ако информацията липсва, използвай null.
В JSON стойностите използвай само обикновени кавички ".
НИКОГА не слагай кавички вътре в стойност — ако трябва да цитираш име, не го ограждай с кавички.

Указания:
- възложител: официалното наименование на ВЪЗЛОЖИТЕЛЯ/КУПУВАЧА на поръчката (организацията, която обявява поръчката). НЕ извличай контролни органи, агенции за справки или институции, посочени само за контакт (напр. НАП, инспекция по труда, министерства) — те НЕ са възложителят.
- предмет: пълното заглавие на поръчката
- прогнозна_стойност: стойността с валутата (напр. "174060.00 EUR")
- начин_на_възлагане: видът процедура, ТОЧНО както е посочен в документа. Възможни видове по ЗОП: "Публично състезание", "Открита процедура", "Ограничена процедура", "Състезателна процедура с договаряне", "Състезателен диалог", "Пряко договаряне", "Договаряне без предварително обявяване". ВАЖНО: "Публично състезание" е ОТДЕЛЕН вид — НЕ го съкращавай до "Състезателна процедура" и НЕ го бъркай със "Състезателна процедура с договаряне" (те са различни). Вземи точния вид от заглавието на решението/обявлението. НЕ предполагай стойност по подразбиране.
- кратко_описание: описанието на услугата, НЕ идентификатори или номера
- критерии_за_подбор: ИЗИСКВАНИЯ КЪМ УЧАСТНИЦИТЕ за допускане (опит, оборот, персонал, сертификати, технически способности) като списък от текстове. В структурираните обявления това е под етикет "Критерии за подбор" / "Описание(BT-750-Lot)" / "BT-809-Lot". ВАЖНО: ако е посочен конкретен критерий (напр. опит с идентични/сходни дейности за последните 3 години), извлечи го дословно. Ако изрично пише "Възложителят не поставя изисквания" — тогава null. ТОВА НЕ СА оценъчните критерии.
- изисквания_за_сертификация: ISO 9001, ISO 27001 или друг конкретен сертификат/стандарт, посочен в документа. Ако няма — null.
- критерии_за_възлагане: ОЦЕНЪЧНИЯТ критерий/критерии за класиране на офертите, като СПИСЪК от текстове, ТОЧНО както са в документа. В структурираните обявления това е под "Критерии за възлагане" / "Вид(BT-539-Lot)". Ако видът е само "Цена" (най-ниска цена печели) — върни само това, напр. ["Цена"]. Ако има няколко показателя с тежести — копирай ги дословно както са изписани. КРИТИЧНО: НЕ измисляй тежести/точки и НЕ пренасяй числа от други поръчки. Копирай САМО каквото реално пише в текущия документ. ТОВА НЕ СА изискванията за подбор.
- краен_срок_за_подаване: КРАЙНАТА ДАТА за подаване/получаване на оферти (конкретна дата, напр. "06-юли-2026"). В структурираните обявления е под етикет "Краен срок за получаване на оферти(BT-131(d)-Lot)" — датата е на следващия ред. ВНИМАНИЕ: това НЕ е "Краен срок за валидност на офертата(BT-98-Lot)" (който е в месеци) и НЕ е срокът за изпълнение на договора.
- дата_на_публикуване: датата на публикуване или изпращане на обявлението (конкретна дата като "08-юни-2026"). В структурираните обявления е под "Дата на изпращане на обявлението(BT-05(a)-notice)". ВНИМАНИЕ: НЕ използвай регистрационни/идентификационни номера (напр. "20260528-0277-0016") като дата.

ВАЖНО ЗА СТРУКТУРИРАНИ ОБЯВЛЕНИЯ (eForms): много обявления са с етикети във формат "Име на полето(BT-код)" и СТОЙНОСТТА е на СЛЕДВАЩИЯ РЕД под етикета. Например:
  Краен срок за получаване на оферти(BT-131(d)-Lot)
  30-юли-2026
Тук стойността е "30-юли-2026". Винаги гледай реда СЛЕД етикета за стойността."""

USER_PROMPT_TEMPLATE = """Анализирай документа и извлечи информацията.

ДОКУМЕНТ: {filename}
---
{text}
---

Върни САМО следния JSON с реална информация. Останалите полета остави null:
{{
  "възложител": null,
  "предмет": null,
  "прогнозна_стойност": null,
  "начин_на_възлагане": null,
  "кратко_описание": null,
  "критерии_за_подбор": null,
  "изисквания_за_сертификация": null,
  "критерии_за_възлагане": null,
  "краен_срок_за_подаване": null,
  "дата_на_публикуване": null
}}"""


FIELDS = [
    "възложител", "предмет", "прогнозна_стойност", "начин_на_възлагане",
    "кратко_описание", "критерии_за_подбор", "изисквания_за_сертификация",
    "критерии_за_възлагане", "краен_срок_за_подаване", "дата_на_публикуване"
]

# Полета с едно правилно значение, повтарящо се в много документи →
# при сливане печели стойността от НАЙ-МНОГО документи (гласуване).
# предмет и кратко_описание се повтарят дословно в обявление/решение/договор.
# критерии_за_възлагане също: ако мнозинството документи казват само "Цена",
# изолирана халюцинация с измислени точки не може да надделее по дължина.
CONSENSUS_FIELDS = {
    "възложител", "начин_на_възлагане", "прогнозна_стойност",
    "краен_срок_за_подаване", "дата_на_публикуване",
    "предмет", "кратко_описание", "критерии_за_възлагане",
}

INVALID_VALUES = {
    "null", "...", "", "не е посочен", "не е намерено",
    "не е посочена", "не е намерена", "n/a", "няма", "липсва",
    "не е посочено", "неизвестно", "не е указано",
}


def is_valid_value(val, field: str = None) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in INVALID_VALUES:
            return False
        # Placeholder: моделът е върнал самото име на полето като стойност,
        # напр. "дата_на_публикуване": "дата на публикуване"
        if field and s == field.replace("_", " ").lower():
            return False
    if isinstance(val, list):
        if len(val) == 0:
            return False
        # Списък само от невалидни елементи също е невалиден
        if all(not is_valid_value(x) for x in val):
            return False
    return True


def normalize_for_vote(val) -> str:
    """Нормализира стойност за броене на гласове (кавички, регистър, интервали)."""
    if isinstance(val, list):
        # За списъци (критерии) — сортираме елементите, за да е стабилен ключът
        parts = sorted(normalize_for_vote(x) for x in val)
        return " | ".join(parts)
    s = str(val).strip().lower()
    for q in ['„', '"', '\u201c', '\u201d', '\u201e', '«', '»', "'"]:
        s = s.replace(q, '')
    s = re.sub(r'\s+', ' ', s)
    return s


# Канонични видове процедури по ЗОП. Ключ = подниз за разпознаване (lowercase),
# стойност = официалното изписване. Подпомага стабилното гласуване за
# начин_на_възлагане, така че различни изписвания да се броят като един глас.
PROCEDURE_TYPES = {
    "публично състезание": "Публично състезание",
    "открита процедура": "Открита процедура",
    "ограничена процедура": "Ограничена процедура",
    "състезателна процедура с договаряне": "Състезателна процедура с договаряне",
    "състезателен диалог": "Състезателен диалог",
    "партньорство за иновации": "Партньорство за иновации",
    "договаряне без предварително обявяване": "Договаряне без предварително обявяване",
    "договаряне без предварителна покана": "Договаряне без предварителна покана",
    "пряко договаряне": "Пряко договаряне",
    "публична покана": "Публична покана",
}

# Ключовете се проверяват от най-специфичния към най-общия, за да не хване
# по-къс подниз преди по-точния (напр. "състезателна процедура с договаряне"
# трябва да се провери преди голото "състезание").
_PROCEDURE_KEYS_SORTED = sorted(PROCEDURE_TYPES, key=len, reverse=True)


def canonicalize_procedure(val):
    """Разпознава вида процедура по ключов подниз и връща официалното изписване."""
    if not isinstance(val, str):
        return val
    low = val.strip().lower()
    for key in _PROCEDURE_KEYS_SORTED:
        if key in low:
            return PROCEDURE_TYPES[key]
    # "публично състезание" понякога се изписва само като "състезание"
    # (недвусмислено). НЕ третираме голото "състезателна процедура" като
    # публично състезание — то може да е начало на "състезателна процедура
    # с договаряне"; там разчитаме на промпта и гласуването.
    if "състезание" in low and "процедура" not in low:
        return "Публично състезание"
    return val


def _vote_key(val) -> str:
    """
    Ключ за групиране на гласове. За дълги текстове (предмет, описание)
    групираме по префикс от първите 60 нормализирани знака, за да не се
    разцепват гласовете заради различни окончания (напр. /ЦУ/ vs
    /Централно управление/). За кратки — пълната нормализирана форма.
    """
    norm = normalize_for_vote(val)
    if len(norm) > 80:
        return norm[:60]
    return norm


def stringify_criterion(item) -> str:
    """
    Превръща критерий в четим стринг. Моделът понякога връща обекти като
    {"име": "...", "тежест": 2} вместо стринг — сглобяваме ги.
    """
    if isinstance(item, dict):
        name = item.get("име") or item.get("name") or item.get("критерий") or ""
        weight = item.get("тежест") or item.get("точки") or item.get("weight")
        name = str(name).strip()
        if weight is not None and str(weight).strip():
            return f"{name} (тежест: {weight})" if name else f"тежест: {weight}"
        return name
    return str(item).strip()


def normalize_criteria_list(val):
    """Нормализира стойност на критерии до списък от стрингове."""
    if isinstance(val, list):
        out = [stringify_criterion(x) for x in val]
        return [s for s in out if s and s.lower() not in INVALID_VALUES]
    if isinstance(val, str):
        return val  # стринг се оставя както е
    return val


def value_richness(val) -> int:
    """Мярка за 'пълнота' на стойност — за избор на по-добрата при конфликт."""
    if isinstance(val, list):
        # Списъците се ценят по сумарна дължина на елементите
        return sum(len(str(x)) for x in val)
    return len(str(val))


def _call_ollama(messages: list[dict]) -> str:
    """Извикване на локален Ollama сървър."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"num_predict": 1000, "temperature": 0.0},
            "messages": messages,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _call_hf(messages: list[dict]) -> str:
    """
    Извикване на HuggingFace Inference Providers през OpenAI-съвместимия router.
    Изисква HF_TOKEN. Моделът се задава чрез HF_MODEL.
    """
    if not HF_TOKEN:
        raise ExtractorError(
            "Липсва HF_TOKEN. Задай го като променлива на средата "
            "(в HuggingFace Space → Settings → Secrets)."
        )
    response = httpx.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "model": HF_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1000,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_model(filename: str, text: str) -> dict:
    """Едно повикване на модела за един чънк текст. Backend-агностично."""
    prompt = USER_PROMPT_TEMPLATE.format(filename=filename, text=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        if LLM_BACKEND == "hf":
            raw = _call_hf(messages)
        else:
            raw = _call_ollama(messages)
        return parse_json_response(raw)
    except httpx.TimeoutException:
        logger.warning(f"  ! Timeout за {filename} — пропускаме чънка")
        return {}
    except httpx.HTTPStatusError as e:
        # Често: 429 (rate limit) или 401 (грешен токен) при HF
        logger.error(f"  ! HTTP грешка за {filename}: {e.response.status_code} {e.response.text[:150]}")
        return {}
    except Exception as e:
        logger.error(f"  ! Грешка за {filename}: {e}")
        return {}


def extract_from_single_doc(filename: str, text: str) -> dict:
    """Разделя документа на чънкове, обработва всеки и слива резултатите."""
    chunks = split_into_chunks(text)
    logger.info(f"  → {filename}: {len(text)} символа, {len(chunks)} чънк(а)"
                f"{' (паралелно ×' + str(MAX_PARALLEL) + ')' if MAX_PARALLEL > 1 and len(chunks) > 1 else ''}")

    def process_one(item):
        i, chunk = item
        result = call_model(filename, chunk)
        if result:
            found = [k for k in FIELDS if is_valid_value(result.get(k), k)]
            logger.info(f"    · чънк {i}/{len(chunks)} ({len(chunk)} символа) ← {found}")
            return result
        logger.info(f"    · чънк {i}/{len(chunks)} ({len(chunk)} символа) ← []")
        return None

    items = list(enumerate(chunks, 1))
    chunk_results = []

    if MAX_PARALLEL > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
            for result in ex.map(process_one, items):
                if result:
                    chunk_results.append(result)
    else:
        for item in items:
            result = process_one(item)
            if result:
                chunk_results.append(result)

    # Сливане на чънкове С гласуване за consensus полета: ако мнозинството
    # чънкове сочат един възложител, изолираното грешно име в един чънк не
    # може да надделее (дори да е по-дълъг низ).
    merged = merge_results(chunk_results, use_voting=True)
    doc_found = [k for k in FIELDS if is_valid_value(merged.get(k), k)]
    logger.info(f"  ← {filename} общо: {doc_found}")
    return merged


def merge_results(results: list[dict], use_voting: bool = False) -> dict:
    """
    Слива списък от резултати.
    - За CONSENSUS_FIELDS (при use_voting=True): печели стойността, срещаща
      се в най-много резултати (гласуване). Така изолирана грешка в един
      документ не надделява над правилната стойност от мнозинството.
    - За останалите полета: печели по-дългата/по-пълната валидна стойност.

    use_voting=True се ползва при сливане на РЕЗУЛТАТИ ОТ ДОКУМЕНТИ (всеки глас
    = един документ). При сливане на чънкове в рамките на един документ
    гласуването е безсмислено, затова там use_voting=False.
    """
    merged = {f: None for f in FIELDS}

    for field in FIELDS:
        # Събираме всички валидни стойности за полето
        candidates = []
        for result in results:
            val = result.get(field)
            if field in ("критерии_за_подбор", "критерии_за_възлагане"):
                val = normalize_criteria_list(val)
            elif field == "начин_на_възлагане":
                val = canonicalize_procedure(val)
            if is_valid_value(val, field):
                candidates.append(val)

        if not candidates:
            continue

        if use_voting and field in CONSENSUS_FIELDS:
            # Гласуване: групираме сходните стойности, печели най-честата група.
            # При равенство — по-дългата оригинална стойност.
            from collections import defaultdict
            groups = defaultdict(list)
            for c in candidates:
                groups[_vote_key(c)].append(c)
            best_key = max(
                groups,
                key=lambda k: (len(groups[k]), max(value_richness(v) for v in groups[k]))
            )
            # сред еднаквите по смисъл взимаме най-дългата изписана форма
            merged[field] = max(groups[best_key], key=value_richness)
        else:
            # По-пълна печели
            best = candidates[0]
            for c in candidates[1:]:
                if value_richness(c) > value_richness(best):
                    best = c
            merged[field] = best

    return merged


def process_documents(filepaths: list[str], url: str = None) -> dict:
    all_results = []

    for fp in filepaths:
        filename = Path(fp).name
        try:
            text = read_document(fp)
            logger.info(f"Прочетен: {filename} ({len(text)} символа)")
            result = extract_from_single_doc(filename, text)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Грешка при {filename}: {e}")

    if not all_results:
        raise ExtractorError("Не може да се извлече информация от нито един документ.")

    final = merge_results(all_results, use_voting=True)
    final["линк"] = url.strip() if url and url.strip() else None

    return final