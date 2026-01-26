#!/usr/bin/env python3
"""Generate the notebooks tables in README.md."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START = "<!-- NOTEBOOKS:START -->"
END = "<!-- NOTEBOOKS:END -->"

SAGEMAKER_BADGE = "https://studiolab.sagemaker.aws/studiolab.svg"
KAGGLE_BADGE = "https://kaggle.com/static/images/open-in-kaggle.svg"
COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

DISPLAY_NAMES: dict[tuple[str, ...], str] = {
    ("causality",): "Causality",
    ("diffusion",): "Diffusion",
    ("implementation",): "Paper implementation",
    ("pytorch",): "Introduce to PyTorch",
    ("pytorch", "audio_classification"): "Audio classification with PyTorch",
    ("pytorch", "computer_vision"): "Computer vision with PyTorch",
    ("pytorch", "natural_language_processing"): "Natural language processing with PyTorch",
}


def repo_slug() -> str:
    env_repo = os.getenv("GITHUB_REPOSITORY")
    if env_repo:
        return env_repo

    config = ROOT / ".git" / "config"
    if config.exists():
        for line in config.read_text().splitlines():
            line = line.strip()
            if not line.startswith("url = "):
                continue
            url = line.split("=", 1)[1].strip()
            # https://github.com/owner/repo.git
            https_match = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", url)
            if https_match:
                return https_match.group(1)
            # git@github.com:owner/repo.git
            ssh_match = re.search(r"git@github\.com:([^/]+/[^/]+?)(?:\.git)?$", url)
            if ssh_match:
                return ssh_match.group(1)

    return "Aisuko/generative-ai"


def discover_notebooks() -> list[Path]:
    notebooks: list[Path] = []
    for path in ROOT.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in path.parts:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        notebooks.append(path)
    return sorted(notebooks, key=lambda p: p.as_posix())


def humanize_title(stem: str) -> str:
    title = stem.replace("_", " ").replace("-", " ").strip()
    title = " ".join(title.split())
    if not title:
        return stem
    return title[0].upper() + title[1:]

def humanize_heading(name: str) -> str:
    title = name.replace("_", " ").replace("-", " ").strip()
    title = " ".join(title.split())
    if not title:
        return name
    return title[0].upper() + title[1:]

def display_heading(parts: tuple[str, ...]) -> str:
    return DISPLAY_NAMES.get(parts, humanize_heading(parts[-1]))


def build_tree(paths: list[Path]) -> dict:
    tree: dict = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        parts = rel.parts
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", []).append(rel)
    return tree


def badge_link(alt: str, badge_url: str, target_url: str) -> str:
    return f"[![{alt}]({badge_url})]({target_url})"


def render_table(files: list[Path], repo: str) -> str:
    header = "|No|Title|Open in SageMaker|Open in Kaggle|Open in Colab|"
    sep = "|---|---|---|---|---|"
    rows: list[str] = [header, sep]
    for idx, rel in enumerate(sorted(files, key=lambda p: p.as_posix()), start=1):
        rel_posix = rel.as_posix()
        encoded_path = quote(rel_posix, safe="/")
        github_blob = f"https://github.com/{repo}/blob/main/{encoded_path}"
        sagemaker_url = f"https://studiolab.sagemaker.aws/import/github/{repo}/blob/main/{encoded_path}"
        kaggle_url = f"https://kaggle.com/kernels/welcome?src={github_blob}"
        colab_url = f"https://colab.research.google.com/github/{repo}/blob/main/{encoded_path}"

        title = humanize_title(rel.stem)
        title_link = f"[{title}]({rel_posix})"
        row = "|".join(
            [
                "",
                str(idx),
                title_link,
                badge_link("Open in SageMaker", SAGEMAKER_BADGE, sagemaker_url),
                badge_link("Kaggle", KAGGLE_BADGE, kaggle_url),
                badge_link("Colab", COLAB_BADGE, colab_url),
                "",
            ]
        )
        rows.append(row)
    return "\n".join(rows)


def render_tree(node: dict, parts: tuple[str, ...], level: int, repo: str) -> list[str]:
    lines: list[str] = []
    heading_level = min(level, 6)
    lines.append(f"{'#' * heading_level} {display_heading(parts)}")
    lines.append("")

    files = node.get("__files__", [])
    if files:
        lines.append(render_table(files, repo))
        lines.append("")

    for child in sorted(k for k in node.keys() if k != "__files__"):
        lines.extend(render_tree(node[child], parts + (child,), level + 1, repo))

    return lines


def render_section(paths: list[Path], repo: str) -> str:
    if not paths:
        return "## Notebooks\n\n_No notebooks found._"

    tree = build_tree(paths)
    lines: list[str] = [
        "## Notebooks",
        "",
        "_Auto-generated from `.ipynb` files. Run `python3 scripts/generate_readme_notebooks.py` to update._",
        "",
    ]

    root_files = tree.get("__files__", [])
    if root_files:
        lines.extend(render_tree({"__files__": root_files}, ("root",), 3, repo))

    for top in sorted(k for k in tree.keys() if k != "__files__"):
        lines.extend(render_tree(tree[top], (top,), 3, repo))

    return "\n".join(lines).rstrip()


def update_readme(section: str) -> None:
    content = README.read_text()
    if START not in content or END not in content:
        raise SystemExit(f"Markers not found in {README}")

    pre, rest = content.split(START, 1)
    _, post = rest.split(END, 1)

    new_block = f"{START}\n{section}\n{END}"
    README.write_text(pre + new_block + post)


def main() -> int:
    repo = repo_slug()
    notebooks = discover_notebooks()
    section = render_section(notebooks, repo)
    update_readme(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
