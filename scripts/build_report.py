from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "reports" / "assets"
STRIP_DIR = ASSETS_DIR / "episode_strips"


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ArialRus", str(regular)))
        pdfmetrics.registerFont(TTFont("ArialRus-Bold", str(bold)))
        return "ArialRus", "ArialRus-Bold"
    return "Helvetica", "Helvetica-Bold"


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullet_list(items: list[str], style: ParagraphStyle) -> list[Paragraph]:
    return [paragraph(f"- {item}", style) for item in items]


def make_episode_strip(gif_path: Path, out_path: Path, frames_count: int = 5) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = imageio.mimread(gif_path)
    if not frames:
        raise RuntimeError(f"GIF has no frames: {gif_path}")

    indices = [
        round(i * (len(frames) - 1) / max(frames_count - 1, 1))
        for i in range(frames_count)
    ]
    pil_frames = [PILImage.fromarray(frames[idx]).convert("RGB") for idx in indices]
    width, height = pil_frames[0].size
    gap = 8
    strip = PILImage.new("RGB", (frames_count * width + (frames_count - 1) * gap, height), "white")
    x = 0
    for frame in pil_frames:
        strip.paste(frame, (x, 0))
        x += width + gap
    strip.save(out_path)
    return out_path


def make_styles(font_regular: str, font_bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            name="TitleRus",
            parent=base["Title"],
            fontName=font_bold,
            fontSize=20,
            leading=24,
            spaceAfter=10,
        ),
        "Heading2": ParagraphStyle(
            name="Heading2Rus",
            parent=base["Heading2"],
            fontName=font_bold,
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "Heading3": ParagraphStyle(
            name="Heading3Rus",
            parent=base["Heading3"],
            fontName=font_bold,
            fontSize=11,
            leading=14,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "BodyText": ParagraphStyle(
            name="BodyRus",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=9.7,
            leading=12.5,
            spaceAfter=5,
        ),
        "Small": ParagraphStyle(
            name="SmallRus",
            parent=base["BodyText"],
            fontName=font_regular,
            fontSize=8.2,
            leading=10.5,
        ),
    }


def add_result_table(story: list, summary: pd.DataFrame, styles: dict[str, ParagraphStyle], font_regular: str, font_bold: str) -> None:
    table_data = [["Метод", "Эпизоды", "Success rate", "Mean return", "Mean steps"]]
    labels = {
        "random": "Random",
        "wm_reward": "World model planning",
        "wm_vlm": "World model + VLM",
    }
    for _, row in summary.iterrows():
        table_data.append(
            [
                labels[row["mode"]],
                f"{int(row['episodes'])}",
                f"{row['success_rate']:.2f}",
                f"{row['mean_return']:.3f}",
                f"{row['mean_steps']:.1f}",
            ]
        )

    table = Table(table_data, colWidths=[6.1 * cm, 2.2 * cm, 3.0 * cm, 3.0 * cm, 2.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
                ("FONTNAME", (0, 0), (-1, 0), font_bold),
                ("FONTNAME", (0, 1), (-1, -1), font_regular),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        paragraph(
            "Оценка проводилась на 30 эпизодах с seeds 0-29. Максимальная длина эпизода - 40 "
            "шагов, horizon планирования - 14, число random-shooting candidates - 768.",
            styles["Small"],
        )
    )


def add_plots(story: list, eval_dir: Path) -> None:
    success_plot = eval_dir / "plots" / "success_rate.png"
    return_plot = eval_dir / "plots" / "mean_return.png"
    if success_plot.exists():
        story.append(Spacer(1, 0.25 * cm))
        story.append(Image(str(success_plot), width=12.3 * cm, height=7.9 * cm))
    if return_plot.exists():
        story.append(Spacer(1, 0.15 * cm))
        story.append(Image(str(return_plot), width=12.3 * cm, height=7.9 * cm))


def add_episode_strips(story: list, styles: dict[str, ParagraphStyle]) -> None:
    gif_specs = [
        ("Random policy", ASSETS_DIR / "random_seed_0.gif", STRIP_DIR / "random_seed_0_strip.png"),
        (
            "World model planning",
            ASSETS_DIR / "wm_reward_seed_0.gif",
            STRIP_DIR / "wm_reward_seed_0_strip.png",
        ),
        (
            "World model + VLM",
            ASSETS_DIR / "wm_vlm_seed_0.gif",
            STRIP_DIR / "wm_vlm_seed_0_strip.png",
        ),
    ]
    for title, gif_path, strip_path in gif_specs:
        if gif_path.exists():
            make_episode_strip(gif_path, strip_path)
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(Image(str(strip_path), width=15.0 * cm, height=3.0 * cm))
            story.append(Spacer(1, 0.12 * cm))


def build_report(eval_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(eval_dir / "summary.csv")
    font_regular, font_bold = register_fonts()
    styles = make_styles(font_regular, font_bold)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.35 * cm,
        title="World Model + VLM Scorer для MiniGrid",
    )
    story = []

    story.append(Paragraph("World Model + VLM Scorer для MiniGrid", styles["Title"]))
    story.append(
        paragraph(
            "В работе реализован небольшой model-based RL проект: агент в MiniGrid планирует "
            "действия через обученную модель мира и использует VLM-based scorer для оценки "
            "будущих состояний относительно текстовой цели <b>agent at the green goal</b>.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("1. Постановка задачи", styles["Heading2"]))
    story.append(
        paragraph(
            "Требовалось объединить world model и VLM-based scorer для управления агентом в "
            "простой среде. Ключевое условие - VLM должен оценивать не только текущее "
            "наблюдение, а именно будущие кадры или состояния, полученные через imagined "
            "rollouts. Также нужно было сравнить метод с random policy и planner-ом без VLM.",
            styles["BodyText"],
        )
    )
    story.extend(
        bullet_list(
            [
                "среда: MiniGrid-Empty-5x5-v0;",
                "world model: компактная RSSM-style модель в духе PlaNet/Dreamer;",
                "VLM scorer: CLIP ViT-B/32 через open_clip;",
                "planner: MPC/random shooting;",
                "метрики: success rate, mean return, mean steps.",
            ],
            styles["BodyText"],
        )
    )

    story.append(Paragraph("2. Общая архитектура", styles["Heading2"]))
    story.append(
        paragraph(
            "Пайплайн состоит из пяти шагов: сбор переходов из MiniGrid, обучение модели мира, "
            "предрасчёт VLM-score для возможных будущих состояний, MPC-планирование по imagined "
            "rollouts и evaluation трёх политик. На каждом реальном шаге среды planner генерирует "
            "candidate action sequences, прокручивает их внутри RSSM и выбирает последовательность "
            "с максимальным objective.",
            styles["BodyText"],
        )
    )
    story.append(
        paragraph(
            "Objective для варианта World model + VLM включает predicted reward, VLM-score и "
            "небольшой goal-distance stabilizer. Последний нужен не как основной метод, а как "
            "практическая стабилизация, потому что CLIP обучался на естественных изображениях, "
            "а MiniGrid является символическим grid-world.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("3. Среда и данные", styles["Heading2"]))
    story.append(
        paragraph(
            "В MiniGrid-Empty-5x5-v0 агент стартует в маленькой комнате и должен дойти до "
            "зелёной клетки-цели. Для обучения world model был собран transition dataset: "
            "перебирались допустимые позиции агента, четыре направления взгляда и три "
            "навигационных действия - left, right, forward. Такой набор покрывает локальную "
            "динамику среды и делает обучение компактной модели стабильным.",
            styles["BodyText"],
        )
    )
    story.append(
        paragraph(
            "Итоговый dataset содержит 3240 переходов. Среди них 60 переходов заканчиваются "
            "достижением цели и дают положительную награду. Помимо этого были сохранены GIF "
            "эпизодов для визуальной проверки поведения политик.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("4. RSSM world model", styles["Heading2"]))
    story.append(
        paragraph(
            "RSSM расшифровывается как Recurrent State-Space Model. В такой модели есть "
            "детерминированная память h_t и stochastic latent state z_t. В полном Dreamer "
            "RSSM обычно работает с пиксельными наблюдениями и latent-представлениями. В этом "
            "проекте сделана компактная state-based версия: она предсказывает следующий "
            "нормализованный state агента, reward и done probability.",
            styles["BodyText"],
        )
    )
    story.append(
        paragraph(
            "Такое упрощение выбрано осознанно: цель задания - показать саму связку world model, "
            "imagined rollouts, VLM-score и MPC. Для маленькой среды state-based RSSM позволяет "
            "получить стабильный результат без долгого обучения pixel decoder-а.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("5. VLM scorer", styles["Heading2"]))
    story.append(
        paragraph(
            "Для VLM scorer используется CLIP ViT-B/32. Все возможные будущие состояния MiniGrid "
            "рендерятся как RGB-кадры и сравниваются с текстом agent at the green goal. Эти "
            "оценки сохраняются в cache, чтобы во время MPC не запускать CLIP заново для каждой "
            "candidate trajectory.",
            styles["BodyText"],
        )
    )
    story.append(
        paragraph(
            "Смысловой момент: score применяется к imagined future states, полученным из rollout-а "
            "world model. То есть planner выбирает действие не только по текущему кадру, а по "
            "оценке возможного будущего.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("6. Планирование", styles["Heading2"]))
    story.append(
        paragraph(
            "Используется MPC с random shooting. На каждом шаге генерируется 768 случайных "
            "последовательностей действий длиной 14. Каждая последовательность прокручивается "
            "внутри learned world model. Затем считается objective, выбирается лучшая "
            "последовательность, в реальной среде выполняется только её первое действие, после "
            "чего процесс повторяется.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("7. Результаты", styles["Heading2"]))
    add_result_table(story, summary, styles, font_regular, font_bold)
    story.append(
        paragraph(
            "Планирование в learned world model заметно лучше random policy. Добавление VLM-score "
            "в этом запуске улучшило success rate с 0.73 до 0.80 и mean return с 0.610 до 0.682. "
            "Также World model + VLM в среднем достигает результата быстрее: 21.1 шага против "
            "24.3 у reward-only planner.",
            styles["BodyText"],
        )
    )
    add_plots(story, eval_dir)

    story.append(PageBreak())
    story.append(Paragraph("8. Визуализация эпизодов", styles["Heading2"]))
    story.append(
        paragraph(
            "Ниже показаны раскадровки трёх GIF-примеров. Они помогают визуально сравнить "
            "поведение random policy, reward-only planner и planner-а с VLM-score. Анимированные "
            "версии GIF также сохранены в репозитории.",
            styles["BodyText"],
        )
    )
    add_episode_strips(story, styles)

    story.append(Paragraph("9. С какими трудностями столкнулись", styles["Heading2"]))
    story.extend(
        bullet_list(
            [
                "CLIP-score оказался шумным на MiniGrid, потому что кадры символические и не похожи на естественные изображения из pretraining.",
                "Полный pixel-level Dreamer был бы слишком тяжёлым для небольшого учебного демо, поэтому была выбрана компактная state-based RSSM.",
                "Random shooting требует много candidate trajectories: меньше candidates работает быстрее, но качество planning падает.",
                "Reward-only planner в простой Empty-среде уже достаточно силён, поэтому VLM-term пришлось взвешивать аккуратно.",
                "Для PDF GIF нельзя встроить как настоящую анимацию, поэтому в отчёт добавлены раскадровки эпизодов.",
            ],
            styles["BodyText"],
        )
    )

    story.append(Paragraph("10. Что можно улучшить дальше", styles["Heading2"]))
    story.extend(
        bullet_list(
            [
                "перейти к DoorKey-8x8, где языковая цель вроде agent next to the key будет полезнее;",
                "обучить pixel decoder и применять VLM напрямую к decoded imagined frames;",
                "заменить random shooting на CEM для более эффективного поиска действий;",
                "добавить uncertainty-aware planning, чтобы planner учитывал неуверенность world model;",
                "сравнить разные VLM и prompt-формулировки.",
            ],
            styles["BodyText"],
        )
    )

    story.append(Paragraph("11. Вывод", styles["Heading2"]))
    story.append(
        paragraph(
            "Проект показывает рабочую связку world model, imagined rollouts, VLM-based scoring "
            "и MPC-planning. Даже в небольшой MiniGrid-среде видно, что learned model позволяет "
            "планировать лучше случайной политики, а добавление VLM-score может улучшить выбор "
            "действий, если аккуратно учитывать шумность vision-language модели на символических "
            "изображениях.",
            styles["BodyText"],
        )
    )

    doc.build(story)
    print(f"Saved report to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PDF report.")
    parser.add_argument("--eval-dir", type=Path, default=Path("artifacts/eval/empty5_tuned"))
    parser.add_argument("--out", type=Path, default=Path("reports/world_model_vlm_minigrid_report.pdf"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(args.eval_dir, args.out)
