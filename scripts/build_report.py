from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def build_report(eval_dir: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(eval_dir / "summary.csv")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="World Model + VLM Scorer for MiniGrid",
    )
    story = []

    story.append(Paragraph("World Model + VLM Scorer for MiniGrid", styles["Title"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        paragraph(
            "This project studies model-based planning in MiniGrid with a compact "
            "RSSM-style world model and a VLM-based goal scorer. The planner samples "
            "candidate action sequences, rolls them out inside the learned world model, "
            "and evaluates imagined future states against the text goal: "
            "<b>agent at the green goal</b>.",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph("Environment and data", styles["Heading2"]))
    story.append(
        paragraph(
            "Environment: MiniGrid-Empty-5x5-v0. The agent starts in a small room and must "
            "reach the green goal cell. Data for the world model was collected from MiniGrid "
            "by enumerating valid agent positions, orientations, and navigation actions "
            "(left, right, forward). This produced 3240 transitions, including 60 goal-reaching "
            "positive-reward transitions.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("World model", styles["Heading2"]))
    story.append(
        paragraph(
            "The world model is a compact Recurrent State-Space Model inspired by PlaNet/Dreamer. "
            "It keeps a deterministic recurrent hidden state h_t and a stochastic latent state z_t. "
            "Given current state and action, it predicts the next normalized state, reward, and done "
            "probability. During planning, candidate action sequences are rolled out inside the "
            "learned model for horizon H=14.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("VLM scorer and planning", styles["Heading2"]))
    story.append(
        paragraph(
            "A CLIP ViT-B/32 model from open_clip was used as the VLM. All possible MiniGrid future "
            "states were rendered as RGB frames and scored against the text goal. During MPC, the "
            "planner applies these scores to imagined future states/frames, not just the current "
            "observation. Random shooting compares 768 candidate action sequences, executes the "
            "first action of the best sequence, and then replans after the environment step.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Baselines", styles["Heading2"]))
    story.append(
        paragraph(
            "The comparison includes three policies: random actions; world-model planning with "
            "predicted reward only; and world-model planning with predicted reward plus VLM score "
            "and a small goal-distance stabilizer. The stabilizer is used because CLIP was trained "
            "primarily on natural images, while MiniGrid frames are highly symbolic.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Quantitative results", styles["Heading2"]))
    table_data = [["Method", "Episodes", "Success rate", "Mean return", "Mean steps"]]
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

    table = Table(table_data, colWidths=[6.3 * cm, 2.2 * cm, 3.0 * cm, 3.0 * cm, 2.5 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
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
            "Evaluation used 30 episodes with seeds 0-29, max episode length 40, horizon 14, "
            "and 768 random-shooting candidates per planning step.",
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
    story.append(Paragraph("Failure modes", styles["Heading2"]))
    story.append(
        paragraph(
            "1. CLIP similarity is noisy on MiniGrid because the rendered frames are symbolic rather "
            "than natural images.\n"
            "2. The compact world model predicts state transitions well in Empty-5x5, but it is not "
            "a full pixel-level Dreamer model and would require a stronger decoder for richer scenes.\n"
            "3. Random shooting is sample-inefficient. Increasing the number of candidates improves "
            "planning quality but also increases CPU runtime.\n"
            "4. Reward-only planning is already strong in this simple environment, so the VLM term "
            "must be weighted carefully.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Future work", styles["Heading2"]))
    story.append(
        paragraph(
            "Further work should include training a pixel decoder and applying the VLM directly to "
            "decoded imagined frames, moving to DoorKey-8x8, replacing random shooting with CEM, "
            "adding uncertainty-aware planning, and testing prompts such as 'agent next to the key' "
            "where language guidance is more informative than a dense reward proxy.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Artifacts", styles["Heading2"]))
    story.append(
        paragraph(
            "The repository includes the final PDF report, result plots, and representative GIFs under "
            "reports/. Training checkpoints, datasets, and evaluation CSV files are reproducible from "
            "the scripts and are written to artifacts/ during local runs.",
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
