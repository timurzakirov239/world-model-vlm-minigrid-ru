# Сообщение для отправки

## Короткая версия

Здравствуйте!

Отправляю решение задания по world model + VLM scorer для управления агентом в MiniGrid.

Репозиторий: https://github.com/timurzakirov239/world-model-vlm-minigrid

PDF-отчёт находится в `reports/world_model_vlm_minigrid_report.pdf`, графики и GIF-примеры - в `reports/assets/`.

С уважением,  
Тимур

## Подробная версия

Здравствуйте!

Отправляю решение задания по объединению world model и VLM-based scorer для управления агентом в простой среде MiniGrid.

Репозиторий: https://github.com/timurzakirov239/world-model-vlm-minigrid

В проекте реализованы:

- среда `MiniGrid-Empty-5x5-v0` и сбор данных;
- компактная RSSM-style world model;
- VLM scorer на основе CLIP для текстовой цели `agent at the green goal`;
- MPC/random-shooting planning по imagined rollouts;
- сравнение с двумя baseline: random policy и world-model planning без VLM.

Основные результаты приведены в README и PDF-отчёте:
`reports/world_model_vlm_minigrid_report.pdf`

В `reports/assets/` также приложены графики и GIF-примеры эпизодов.

С уважением,  
Тимур
