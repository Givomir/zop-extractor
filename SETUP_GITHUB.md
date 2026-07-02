# Настройка: GitHub → HuggingFace Space (автоматичен deploy)

Това ръководство те води през свързването на GitHub repo с HuggingFace Space,
така че при всеки `git push` Space-ът да се обновява автоматично.

**Не ти трябва Dockerfile** — използваме Gradio SDK, който HuggingFace построява
автоматично от `requirements.txt`.

---

## Какви файлове влизат в repo-то

```
zop-extractor/
├── app.py                      # Gradio интерфейс
├── extractor.py                # логиката за извличане
├── requirements.txt            # зависимости
├── README.md                   # с YAML header за HF Space
├── .gitignore
└── .github/
    └── workflows/
        └── sync-to-hub.yml     # GitHub Action за auto-deploy
```

(Тестовите файлове `test_*.py` са изключени чрез `.gitignore` — ако искаш да ги
качиш за документация, махни съответния ред от `.gitignore`.)

---

## Стъпка 1 — Създай HuggingFace Space

1. Иди на https://huggingface.co/new-space
2. Име: например `zop-extractor`
3. SDK: избери **Gradio**
4. Видимост: Public (за да тестват хората) или Private
5. Натисни **Create Space** (остави го празен засега)

## Стъпка 2 — Вземи HuggingFace токен

1. Иди на https://huggingface.co/settings/tokens
2. **New token** → тип **Write** (или fine-grained с достъп до твоя Space)
3. Копирай токена (започва с `hf_...`)

## Стъпка 3 — Създай GitHub repo

1. Създай нов repo на https://github.com/new (напр. `zop-extractor`)
2. Качи файловете (през уеб интерфейса или с git):

```bash
git init
git add .
git commit -m "Първоначална версия на ЗОП екстрактора"
git branch -M main
git remote add origin https://github.com/ТВОЕТО_ИМЕ/zop-extractor.git
git push -u origin main
```

## Стъпка 4 — Добави HF_TOKEN като GitHub secret

1. В GitHub repo-то: **Settings** → **Secrets and variables** → **Actions**
2. **New repository secret**
3. Име: `HF_TOKEN`
4. Стойност: токенът от Стъпка 2
5. **Add secret**

## Стъпка 5 — Настрой workflow файла

Отвори `.github/workflows/sync-to-hub.yml` и замени реда:

```yaml
huggingface_repo_id: ВАШЕТО_HF_ИМЕ/zop-extractor
```

с твоето HuggingFace потребителско име и име на Space, например:

```yaml
huggingface_repo_id: ivan/zop-extractor
```

Commit-ни промяната. **Това автоматично стартира workflow-а** и качва всичко в Space-а.

## Стъпка 6 — Добави HF_TOKEN и в самия Space

Workflow-ът качва кода, но приложението има нужда от `HF_TOKEN` по време на работа
(за да вика модела). Това е ОТДЕЛЕН secret, в HuggingFace, не в GitHub:

1. В Space-а: **Settings** → **Variables and secrets**
2. **New secret**
3. Име: `HF_TOKEN`, Стойност: същият токен
4. (По желание) добави `HF_MODEL` ако искаш друг модел от подразбирания

---

## Как работи след това

- Промениш код локално → `git push` → GitHub Action се пуска → Space се обновява за ~1-2 мин.
- Можеш да пуснеш deploy и ръчно: GitHub repo → **Actions** → избери workflow → **Run workflow**.

## Чести проблеми

- **Workflow гръмва с „authentication failed":** провери че `HF_TOKEN` в GitHub
  secrets е правилен и от тип Write.
- **Space се build-ва, но приложението дава грешка за липсващ токен:** не си
  добавил `HF_TOKEN` в Space secrets (Стъпка 6) — това е различно от GitHub secret.
- **Файлове над 10MB:** изискват Git-LFS. Този проект няма такива, но не качвай
  примерни документи над 10MB без LFS.
- **429 / rate limit от модела:** безплатният HF tier има месечни лимити.
  При интензивно ползване обмисли PRO акаунт или намали `LLM_PARALLEL`.
