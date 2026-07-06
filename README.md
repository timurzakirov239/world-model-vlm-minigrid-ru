# World Model + VLM Scorer для MiniGrid

Проект выполнен в рамках задания по объединению **world model** и
**VLM-based scorer** для управления агентом в простой среде.

В качестве среды используется `MiniGrid-Empty-5x5-v0`: агент находится в
маленькой комнате и должен дойти до зелёной цели. Вместо того чтобы выбирать
действия только случайно, агент строит несколько возможных будущих траекторий
внутри обученной модели мира, оценивает их и выполняет первый шаг из лучшего
плана.

Текстовая цель для VLM:

```text
agent at the green goal
```

## Что Есть В Репозитории

- код среды и подготовки данных;
- компактная RSSM-style world model в духе PlaNet/Dreamer;
- VLM scorer на основе CLIP;
- MPC/random-shooting planner по imagined rollouts;
- сравнение с двумя baseline;
- PDF-отчёт, графики и GIF-примеры эпизодов.

Основной отчёт находится здесь:

```text
reports/world_model_vlm_minigrid_report.pdf
```

Визуализации:

```text
reports/assets/success_rate.png
reports/assets/mean_return.png
reports/assets/random_seed_0.gif
reports/assets/wm_reward_seed_0.gif
reports/assets/wm_vlm_seed_0.gif
```

В PDF-отчёт также добавлены раскадровки этих GIF-примеров, чтобы визуальная
часть была видна прямо в документе.

## Метод

### Среда

MiniGrid - это простая grid-world среда для задач reinforcement learning. В
использованной версии агент стартует в комнате и должен добраться до зелёной
клетки-цели.

### World Model

Модель мира обучается предсказывать следующий state, reward и done по текущему
state и действию:

```text
state_t + action_t -> state_t+1, reward_t+1, done_t+1
```

В проекте используется компактная RSSM-style модель: она содержит рекуррентное
скрытое состояние и latent-переменную. Это упрощённая версия идеи PlaNet/Dreamer,
достаточная для демонстрации imagined rollouts в небольшой среде.

### VLM Scorer

Для оценки будущих состояний используется CLIP ViT-B/32. Возможные будущие
состояния рендерятся как кадры MiniGrid и сравниваются с текстовой целью
`agent at the green goal`.

Важно: VLM-score применяется именно к будущим состояниям из rollout-а, а не
только к текущему наблюдению.

### Планирование

Планирование реализовано через MPC/random shooting:

1. генерируется много кандидатных последовательностей действий;
2. каждая последовательность прокручивается внутри world model;
3. imagined future states оцениваются через reward и VLM-score;
4. выбирается лучшая последовательность;
5. в настоящей среде выполняется только первое действие;
6. на следующем шаге планирование повторяется.

## Baselines

Сравниваются три варианта:

| Метод | Описание |
| --- | --- |
| Random | случайные действия |
| World-model planning | планирование в world model без VLM-score |
| World-model planning + VLM | планирование в world model с VLM-score по будущим кадрам |

## Результаты

Оценка проводилась на 30 эпизодах, seeds `0-29`.

| Метод | Success rate | Mean return | Mean steps |
| --- | ---: | ---: | ---: |
| Random | 0.40 | 0.307 | 34.3 |
| World-model planning | 0.73 | 0.610 | 24.3 |
| World-model planning + VLM | 0.80 | 0.682 | 21.1 |

По результатам видно, что планирование в модели мира заметно лучше случайной
политики, а добавление VLM-score в данном эксперименте улучшает success rate и
средний return.

## Структура Проекта

```text
src/            Основной код: среда, модель мира, scorer, planner
scripts/        Скрипты для сбора данных, обучения и оценки
reports/        PDF-отчёт, графики и GIF-примеры
configs/        Папка под конфиги экспериментов
artifacts/      Локальные результаты запусков, не добавляются в git
```

## Как Запустить Эксперимент

Установка зависимостей:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Минимальный порядок запуска:

```powershell
.\.venv\Scripts\python.exe scripts\collect_transition_dataset.py
.\.venv\Scripts\python.exe scripts\train_world_model.py
.\.venv\Scripts\python.exe scripts\build_vlm_cache.py
.\.venv\Scripts\python.exe scripts\evaluate_agents.py
```

После запуска результаты сохраняются в папку `artifacts/`.


