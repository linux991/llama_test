
# Инструкция for dummies: как повторить эксперимент на хоррор-историях

## Что делает notebook

Notebook повторяет основную логику статьи *When Detection Fails: The Power of Fine-Tuned Models to Generate Human-Like Social Media Text*, но переносит её с коротких соцсетевых постов на англоязычные хоррор-истории.

Оригинальный эксперимент строился так: авторы собирали человеческие тексты, генерировали AI-тексты разными стратегиями, дообучали модели через QLoRA, сравнивали human/base/fine-tuned тексты и проверяли, насколько хорошо детекторы отличают AI-текст от человеческого. В статье использовались GPT-4o, GPT-4o-mini, Llama-3-8B-Instruct и Llama-3.2-1B-Instruct; для Llama применялись QLoRA, 4-bit quantization, LoRA rank 64, alpha 16 и 5 эпох обучения. В notebook эта схема адаптирована под horror corpus и фиксируется на `Meta-Llama-3-8B-Instruct`.

## Что именно берём из статьи

Для нашей репликации оставляем не всю статью, а её ключевой дизайн:

- human-written corpus: вместо Twitter/X берём англоязычные creepypasta/horror stories;
- generation strategy: используем аналог `Generate From Topic`, то есть сначала делаем краткое описание фрагмента, затем просим модель написать новый horror fragment по описанию;
- base condition: генерируем тексты базовой `Meta-Llama-3-8B-Instruct`;
- fine-tuned condition: дообучаем ту же Llama 3 8B через QLoRA и снова генерируем по тем же описаниям;
- comparison: считаем стилометрию и обучаем supervised detector для `human vs base AI` и `human vs fine-tuned AI`;
- главный ожидаемый эффект: fine-tuned AI должен быть ближе к human по признакам и труднее для детектора.

## Что тебе нужно подготовить

1. Собери `.txt` файлы с хоррор-историями.
2. Проверь лицензию текстов. Лучше использовать public domain, CC BY, CC BY-SA, CC0 или тексты с разрешением автора.
3. Сохрани все `.txt` в одну папку.
4. Заархивируй папку в `horror_texts.zip`.

Пример:

```text
horror_texts.zip
├── story_001.txt
├── story_002.txt
├── story_003.txt
└── story_004.txt
```

## Если используешь Kaggle-датасет 3500 Popular Creepypastas

Для датасета `thomaskonstantin/3500-popular-creepypastas` добавлен отдельный скрипт:

```bash
python3 scripts/extract_creepypastas.py \
  --input creepypastas.xlsx \
  --output-dir data/creepypasta_stories_txt \
  --overwrite
```

Что делает скрипт:

- принимает Kaggle ZIP, распакованную папку или отдельный XLSX/CSV/JSON/TXT;
- автоматически ищет колонку с текстом истории;
- чистит HTML и лишние пробелы;
- удаляет пустые, слишком короткие и дублирующиеся записи;
- сохраняет каждую историю отдельным `.txt` файлом;
- создаёт `metadata.csv` со списком извлечённых историй.

После запуска получится папка:

```text
data/creepypasta_stories_txt/
├── 0001_story_title.txt
├── 0002_another_story.txt
└── metadata.csv
```

Если скрипт не угадал колонку с текстом, укажи её явно:

```bash
python3 scripts/extract_creepypastas.py \
  --input creepypastas.xlsx \
  --text-column body \
  --title-column story_name \
  --output-dir data/creepypasta_stories_txt \
  --overwrite
```

Эта папка лежит внутри `data/`, поэтому не попадёт в git. Для Colab её можно заархивировать и загрузить как обычный `horror_texts.zip`.

## Как запустить

1. Открой Google Colab.
2. Нажми `File → Upload notebook`.
3. Загрузи файл `horror_qlora_detection_experiment_colab.ipynb`.
4. В меню выбери `Runtime → Change runtime type`.
5. В поле `Hardware accelerator` выбери `T4 GPU`.
6. Запускай ячейки сверху вниз.

## Где нужно что-то менять

Главный блок называется `CONFIG`.

Эксперимент теперь зафиксирован как English-only + Llama 3 8B:

```python
"LANGUAGE": "en"
"GEN_MODEL_NAME": "meta-llama/Meta-Llama-3-8B-Instruct"
"DETECTOR_MODEL_NAME": "openai-community/roberta-base-openai-detector"
"MAX_HUMAN_SAMPLES": 1000
"NUM_EPOCHS": 5
```

Для smoke test можно временно ослабить запуск:

```python
"MAX_HUMAN_SAMPLES": 200
"NUM_EPOCHS": 1
```

Для Llama 3 8B нужен Hugging Face доступ:

1. Открой страницу `meta-llama/Meta-Llama-3-8B-Instruct` на Hugging Face.
2. Прими лицензию Meta.
3. В Colab добавь `HF_TOKEN` в secrets или запусти ячейку `notebook_login()`.

## Что происходит по шагам

### Шаг 1. Установка библиотек

Notebook устанавливает библиотеки для:
- загрузки моделей;
- QLoRA fine-tuning;
- обработки датасетов;
- обучения детектора;
- расчёта метрик.

### Шаг 2. Загрузка корпуса

Ты загружаешь ZIP с `.txt` файлами. Notebook распаковывает его и ищет все тексты.

### Шаг 3. Нарезка текстов

Длинные рассказы режутся на фрагменты. Это нужно, чтобы сравнивать тексты сопоставимой длины.

По умолчанию:

```python
"MIN_CHARS": 500
"MAX_CHARS": 1200
```

То есть каждый пример будет примерно от 500 до 1200 символов.

### Шаг 4. Создание описаний

Для каждого человеческого фрагмента создаётся короткое задание для генерации. Например:

```text
Напиши короткий фрагмент хоррор-истории в художественном стиле.
Сохрани атмосферу тревоги, неизвестности и нарастающего страха.
Опорные мотивы/слова: дверь, ночь, тень, голос...
```

Это аналог “Topic” из оригинальной статьи.

### Шаг 5. Генерация base AI

Базовая модель генерирует хоррор-фрагменты до дообучения.

Это условие нужно, чтобы понять, насколько “обычная” модель отличается от человеческих текстов.

### Шаг 6. QLoRA fine-tuning

Модель дообучается на парах:

```text
описание → человеческий хоррор-фрагмент
```

То есть она учится писать в стиле твоего корпуса.

### Шаг 7. Генерация fine-tuned AI

После обучения та же модель снова генерирует тексты по тем же описаниям.

Теперь можно сравнить:

```text
human vs base AI
human vs fine-tuned AI
```

### Шаг 8. Стилометрический анализ

Notebook считает признаки:

- длина текста;
- число слов;
- type-token ratio;
- доля заглавных букв;
- восклицательные знаки;
- вопросительные знаки;
- многоточия;
- кавычки / признаки прямой речи;
- частотность слов страха.

Главная идея: если fine-tuned AI ближе к human по этим признакам, значит дообучение сработало.

### Шаг 9. Детектор

Notebook обучает простой supervised detector:

1. human vs base AI;
2. human vs fine-tuned AI.

Если accuracy/F1 ниже во втором случае, значит fine-tuned тексты труднее отличить от человеческих.

## Какие файлы получатся

После запуска будут созданы:

```text
horror_experiment/
├── data/
│   ├── human_horror_chunks.csv
│   ├── human_with_descriptions.csv
│   ├── ai_base_generations.csv
│   └── ai_finetuned_generations.csv
├── outputs/
│   ├── stylometric_features.csv
│   ├── rank_biserial_effects.csv
│   ├── detector_summary.csv
│   └── mixed_examples_for_manual_review.csv
└── models/
    ├── horror_qlora_adapter_final/
    ├── detector_human_vs_base_ai/
    └── detector_human_vs_finetuned_ai/
```

Самые важные файлы:

- `ai_base_generations.csv` — тексты базовой модели;
- `ai_finetuned_generations.csv` — тексты после дообучения;
- `rank_biserial_effects.csv` — различия между human и AI;
- `detector_summary.csv` — итоговые метрики детектора;
- `mixed_examples_for_manual_review.csv` — примеры для ручного анализа.

## Как интерпретировать результат

Тебе нужны два основных вывода.

### Вывод 1

Если стилометрические различия между human и fine-tuned AI меньше, чем между human и base AI, значит fine-tuning сделал модель ближе к человеческому корпусу.

Смотри файл:

```text
rank_biserial_effects.csv
```

Чем ближе `rank_biserial` к нулю, тем меньше различие.

### Вывод 2

Если детектор хуже отличает fine-tuned AI от human, чем base AI от human, значит эксперимент повторяет основную идею статьи.

Смотри файл:

```text
detector_summary.csv
```

Пример желаемой картины:

```text
human vs base AI          accuracy = 0.90
human vs fine-tuned AI    accuracy = 0.70
```

Это значит, что после fine-tuning AI-текст стал менее обнаружимым.

## Что написать в работе

Можно использовать такую формулировку:

> В практической части была реализована адаптированная репликация эксперимента Dawkins et al. по оценке обнаружимости текстов, созданных базовой и дообученной языковой моделью. В отличие от оригинального исследования, материалом стали фрагменты хоррор-историй. Корпус человеческих текстов был очищен и сегментирован на фрагменты сопоставимой длины. Для каждого фрагмента автоматически формировалось краткое генеративное описание, после чего базовая instruction-модель создавала синтетические хоррор-фрагменты. Затем модель была дообучена методом QLoRA на парах “описание — человеческий фрагмент”, после чего генерация была повторена. Полученные тексты сравнивались со стилометрическими признаками человеческого корпуса, а также оценивались с помощью supervised PLM-детектора.

## Важные ограничения

Это не полная копия статьи, а адаптированная репликация.

Отличия:
- вместо Twitter/X используются художественные хоррор-фрагменты;
- вместо политических topic/stance используется horror description;
- off-the-shelf детекторы для художественного horror-домена ненадёжны;
- основной детектор обучается внутри эксперимента;
- качество зависит от лицензий, объёма и чистоты корпуса.

## Частые ошибки

### Ошибка: CUDA out of memory

Что делать:
- уменьшить `MAX_HUMAN_SAMPLES`;
- поставить `LORA_R = 16`;
- уменьшить `MAX_NEW_TOKENS`;
- использовать модель 1B–1.5B, а не 7B/8B.

### Ошибка: нет GPU

Что делать:
- `Runtime → Change runtime type → T4 GPU`.

### Ошибка с Llama access

Llama-модели требуют принятия лицензии на Hugging Face. Для текущего эксперимента не заменяй модель на Qwen: нужно принять лицензию `meta-llama/Meta-Llama-3-8B-Instruct` и авторизоваться через `HF_TOKEN` или `notebook_login()`.

### Генерации слишком короткие или плохие

Увеличь:

```python
"MAX_NEW_TOKENS": 500
```

И проверь качество исходных `.txt`.

### Модель копирует исходные тексты

Это риск при маленьком корпусе. Что делать:
- увеличить корпус;
- уменьшить число эпох;
- вручную проверить `mixed_examples_for_manual_review.csv`;
- добавить проверку на near-duplicate.
