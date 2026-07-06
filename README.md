# World Model + VLM Scorer для MiniGrid

Проект демонстрирует управление агентом в простой среде MiniGrid с помощью
связки из двух компонентов:

- **world model** - модель мира, которая учится предсказывать последствия действий;
- **VLM scorer** - оценщик на основе vision-language model, который сравнивает будущие кадры с текстовой целью.

Текстовая цель:

```text
agent at the green goal
```

Идея эксперимента: агент не просто выбирает случайное действие, а сначала
"воображает" несколько возможных будущих траекторий внутри обученной модели мира,
оценивает эти будущие состояния и выполняет первый шаг из лучшего плана.

## Что Реализовано

- Среда: `MiniGrid-Empty-5x5-v0`
- Модель мира: компактная RSSM-style модель в духе PlaNet/Dreamer
- VLM scorer: CLIP ViT-B/32 через `open_clip_torch`
- Планирование: MPC/random shooting по imagined rollouts
- Горизонт планирования: 14 шагов
- Число кандидатных последовательностей действий: 768

## Коротко О Методе

### Среда

MiniGrid - это маленький grid-world: агент находится в комнате, может
поворачиваться и идти вперёд. В выбранной среде цель агента - дойти до зелёной
клетки.

### World Model

World model обучается по переходам вида:

```text
состояние_t + действие_t -> состояние_t+1, reward_t+1, done_t+1
```

В проекте используется компактная RSSM-style модель. Она хранит скрытое
рекуррентное состояние и latent-переменную, а затем предсказывает следующий
нормализованный state агента, награду и вероятность завершения эпизода.

### VLM Scorer

VLM scorer использует CLIP: модель получает изображение и текстовую цель, после
чего выдаёт численную оценку их похожести. Важно, что score применяется не
только к текущему наблюдению, а к будущим состояниям, полученным через imagined
rollouts в world model.

### Планирование

На каждом шаге агент:

1. Берёт текущее состояние среды.
2. Случайно генерирует много последовательностей действий.
3. Прогоняет эти действия внутри world model.
4. Получает imagined future states.
5. Оценивает их через reward/VLM objective.
6. Выполняет первое действие из лучшей последовательности.
7. Повторяет планирование на следующем шаге.

## Baselines

Сравниваются три политики:

| Метод | Описание |
| --- | --- |
| Random | Случайные действия |
| World-model planning | MPC в модели мира только по предсказанной награде |
| World-model planning + VLM | MPC с предсказанной наградой и VLM-score по будущим кадрам |

## Результаты

Оценка проводилась на 30 эпизодах, seeds `0-29`.

| Метод | Success rate | Mean return | Mean steps |
| --- | ---: | ---: | ---: |
| Random | 0.40 | 0.307 | 34.3 |
| World-model planning | 0.73 | 0.610 | 24.3 |
| World-model planning + VLM | 0.80 | 0.682 | 21.1 |

## Структура Репозитория

```text
src/            Код среды, RSSM-модели, scorer-а и planner-а
scripts/        Скрипты для сбора данных, обучения, оценки и отчёта
reports/        PDF-отчёт, графики и GIF-примеры
configs/        Папка под конфиги экспериментов
artifacts/      Локальные генерируемые файлы, не добавляются в git
```

## Основные Файлы

- PDF-отчёт: `reports/world_model_vlm_minigrid_report.pdf`
- Графики:
  - `reports/assets/success_rate.png`
  - `reports/assets/mean_return.png`
- GIF-примеры:
  - `reports/assets/random_seed_0.gif`
  - `reports/assets/wm_reward_seed_0.gif`
  - `reports/assets/wm_vlm_seed_0.gif`

## Как Запустить

Создать виртуальное окружение и установить зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Собрать transition dataset:

```powershell
.\.venv\Scripts\python.exe scripts\collect_transition_dataset.py --env-id MiniGrid-Empty-5x5-v0 --repeats 30 --seed 0 --gif
```

Обучить RSSM world model:

```powershell
.\.venv\Scripts\python.exe scripts\train_world_model.py --epochs 300 --batch-size 256 --lr 0.003
```

Построить VLM-score cache:

```powershell
.\.venv\Scripts\python.exe scripts\build_vlm_cache.py --env-id MiniGrid-Empty-5x5-v0 --goal-text "agent at the green goal"
```

Запустить сравнение трёх политик:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_agents.py --seeds "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29" --max-steps 40 --horizon 14 --candidates 768 --vlm-weight 0.25 --reward-weight 1.0 --distance-weight 0.5 --out-dir artifacts\eval\empty5_tuned
```

Пересобрать графики и PDF-отчёт:

```powershell
.\.venv\Scripts\python.exe scripts\make_result_plots.py --eval-dir artifacts\eval\empty5_tuned
.\.venv\Scripts\python.exe scripts\build_report.py --eval-dir artifacts\eval\empty5_tuned --out reports\world_model_vlm_minigrid_report.pdf
```

## Ограничения

- CLIP обучался в основном на естественных изображениях, поэтому на символических
  MiniGrid-картинках его score шумный.
- Используется компактная state-based RSSM, а не полный pixel-level Dreamer.
- Random shooting прост в реализации, но менее эффективен, чем CEM.
