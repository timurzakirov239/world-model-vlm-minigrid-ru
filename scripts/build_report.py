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


def build_report(eval_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(eval_dir / "summary.csv")
    font_regular, font_bold = register_fonts()

    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            name="TitleRus",
            parent=base_styles["Title"],
            fontName=font_bold,
            fontSize=20,
            leading=24,
            spaceAfter=10,
        ),
        "Heading2": ParagraphStyle(
            name="Heading2Rus",
            parent=base_styles["Heading2"],
            fontName=font_bold,
            fontSize=14,
            leading=18,
            spaceBefore=8,
            spaceAfter=6,
        ),
        "Heading3": ParagraphStyle(
            name="Heading3Rus",
            parent=base_styles["Heading3"],
            fontName=font_bold,
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "BodyText": ParagraphStyle(
            name="BodyRus",
            parent=base_styles["BodyText"],
            fontName=font_regular,
            fontSize=10,
            leading=13,
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            name="SmallRus",
            parent=base_styles["BodyText"],
            fontName=font_regular,
            fontSize=8.5,
            leading=11,
        ),
    }

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="World Model + VLM Scorer для MiniGrid",
    )
    story = []

    story.append(Paragraph("World Model + VLM Scorer для MiniGrid", styles["Title"]))
    story.append(
        paragraph(
            "Цель проекта - показать небольшую model-based RL систему, где агент в MiniGrid "
            "планирует действия через обученную модель мира и использует VLM-based scorer "
            "для оценки будущих состояний относительно текстовой цели: "
            "<b>agent at the green goal</b>.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Среда и данные", styles["Heading2"]))
    story.append(
        paragraph(
            "В качестве среды используется MiniGrid-Empty-5x5-v0. Агент стартует в небольшой "
            "комнате и должен дойти до зелёной клетки-цели. Данные для модели мира собраны "
            "из MiniGrid путём перебора допустимых позиций агента, направлений взгляда и "
            "действий навигации: left, right, forward. Всего получено 3240 переходов, включая "
            "60 переходов с положительной наградой за достижение цели.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Модель мира", styles["Heading2"]))
    story.append(
        paragraph(
            "Модель мира реализована как компактная RSSM-style модель в духе PlaNet/Dreamer. "
            "Она хранит рекуррентное скрытое состояние h_t и stochastic latent state z_t. "
            "По текущему состоянию и действию модель предсказывает следующий нормализованный "
            "state агента, reward и вероятность done. Во время планирования candidate action "
            "sequences прокручиваются внутри этой learned world model на горизонте H=14.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("VLM scorer и планирование", styles["Heading2"]))
    story.append(
        paragraph(
            "Для VLM scorer используется CLIP ViT-B/32 из open_clip. Возможные будущие состояния "
            "рендерятся как RGB-кадры MiniGrid и сравниваются с текстовой целью. Важно, что "
            "VLM-score применяется к imagined future states из rollout-а, а не только к текущему "
            "наблюдению. Планировщик использует MPC/random shooting: на каждом шаге сравнивает "
            "768 кандидатных последовательностей действий, выполняет первое действие из лучшей "
            "последовательности и затем планирует заново.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Baseline-сравнение", styles["Heading2"]))
    story.append(
        paragraph(
            "Сравниваются три варианта: random policy, planning в world model без VLM-score и "
            "planning в world model с VLM-score. В последнем варианте также используется небольшой "
            "goal-distance stabilizer, потому что CLIP обучался в основном на естественных "
            "изображениях, а MiniGrid-кадры являются символическими.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Количественные результаты", styles["Heading2"]))
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

    success_plot = eval_dir / "plots" / "success_rate.png"
    return_plot = eval_dir / "plots" / "mean_return.png"
    if success_plot.exists():
        story.append(Spacer(1, 0.3 * cm))
        story.append(Image(str(success_plot), width=12.5 * cm, height=8.1 * cm))
    if return_plot.exists():
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(return_plot), width=12.5 * cm, height=8.1 * cm))

    story.append(PageBreak())
    story.append(Paragraph("Визуализация эпизодов", styles["Heading2"]))
    story.append(
        paragraph(
            "Ниже показаны раскадровки трёх GIF-примеров из reports/assets/. В самом репозитории "
            "эти эпизоды сохранены как анимированные GIF.",
            styles["BodyText"],
        )
    )

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
            story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("Основные проблемы", styles["Heading2"]))
    story.append(
        paragraph(
            "1. CLIP-score на MiniGrid шумный, потому что изображения символические и сильно "
            "отличаются от естественных изображений из pretraining.\n"
            "2. Компактная RSSM хорошо подходит для Empty-5x5, но это не полный pixel-level "
            "Dreamer. Для более сложных сред нужен stronger decoder и более богатое latent state.\n"
            "3. Random shooting прост и нагляден, но sample-inefficient: рост числа candidates "
            "улучшает планирование, но увеличивает время работы на CPU.\n"
            "4. В простой среде reward-only planning уже достаточно силён, поэтому VLM-term "
            "требует аккуратного веса.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Что можно улучшить дальше", styles["Heading2"]))
    story.append(
        paragraph(
            "Дальше логично перейти к DoorKey-8x8, обучить pixel decoder и применять VLM напрямую "
            "к decoded imagined frames. Также можно заменить random shooting на CEM, добавить "
            "uncertainty-aware planning и проверить цели вроде 'agent next to the key', где "
            "языковая цель может быть полезнее простой reward-эвристики.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Файлы", styles["Heading2"]))
    story.append(
        paragraph(
            "PDF-отчёт, графики и GIF-примеры лежат в reports/. Данные, checkpoints и evaluation "
            "CSV воспроизводятся скриптами и сохраняются локально в artifacts/.",
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
