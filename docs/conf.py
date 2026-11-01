"""小鸭子 MiniDuck 使用手册的 Sphinx 配置。版本信息只从 main 分支的 versions.toml 读（发布时取到 _main/）。"""

import tomllib
from pathlib import Path

BASE = "https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course"
versions = tomllib.loads((Path(__file__).parent / "_main/versions.toml").read_text(encoding="utf-8"))


def asset(entry: dict) -> str:
    return f"{BASE}/releases/download/{entry['release']}/{entry['file']}"


project = "小鸭子 MiniDuck 使用手册"
author = "Xbotics 具身智能社区"
language = "zh_CN"
root_doc = "index"
source_suffix = {".md": "markdown"}

extensions = ["myst_parser", "sphinx_copybutton"]
exclude_patterns = ["_build", "_main"]

myst_enable_extensions = ["attrs_inline", "colon_fence", "html_image", "substitution"]
myst_heading_anchors = 3
myst_substitutions = {
    "course_version": versions["course"]["version"],
    "hardware": f"`{versions['hardware']['id']}`",
    "hardware_json": f"`sim_model/hardware/{versions['hardware']['id']}.json`",
    "policy_dir": f"`policies/{versions['hardware']['id']}/<模型>/`",
    "servo": versions["hardware"]["servo"],
    "board": versions["hardware"]["board"],
    "firmware": versions["firmware"]["version"],
    "firmware_link": f"[`{versions['firmware']['dir']}/`]({BASE}/tree/main/{versions['firmware']['dir']})",
    "firmware_dir": f"`{versions['firmware']['dir']}/`",
    "tool_link": f"[{versions['flash_tool']['file']}]({asset(versions['flash_tool'])})",
}
copybutton_prompt_text = "$ "

html_theme = "sphinx_book_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["site.css"]
html_context = {"default_mode": "auto"}
html_theme_options = {
    "repository_url": BASE,
    "repository_branch": "docs/github-pages",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
}
