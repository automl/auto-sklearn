#!/usr/bin/env python3
"""Generate API documentation for auto-sklearn."""

import os
import re
from pathlib import Path
from typing import List

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Configure the paths
SRC = Path(__file__).parent.parent / "autosklearn"
DOCS = Path(__file__).parent
API_DIR = DOCS / "api"

# Create API directory if it doesn't exist
API_DIR.mkdir(exist_ok=True)

# Patterns to exclude from documentation
EXCLUDE_PATTERNS = [
    r"^.*__pycache__.*$",
    r"^.*\.(pyc|pyo)$",
    r"^.*test.*$",
    r"^.*automl_common.*$",
    r"^.*metalearning/files.*$",
    r"^.*experimental/balanced_accuracy.*$",
    r"^.*experimental/log_loss.*$",
    r"^.*experimental/roc_auc.*$",
]

def should_include(path: Path) -> bool:
    """Check if a file should be included in the documentation."""
    path_str = str(path)
    
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, path_str, re.IGNORECASE):
            return False
    
    # Only include Python files
    if path.suffix != ".py":
        return False
    
    # Exclude __init__.py that are empty
    if path.name == "__init__.py":
        content = path.read_text()
        if len(content.strip()) < 10:
            return False
    
    return True

def get_module_name(path: Path) -> str:
    """Convert a file path to a module name."""
    rel_path = path.relative_to(SRC.parent)
    return str(rel_path).replace("/", ".").replace(".py", "")

# Walk through the source directory
for path in sorted(SRC.rglob("*.py")):
    if not should_include(path):
        continue
    
    module_path = path.relative_to(SRC.parent).with_suffix("")
    doc_path = path.relative_to(SRC).with_suffix(".md")
    full_doc_path = Path("api", doc_path)
    
    parts = list(module_path.parts)
    
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue
    
    nav[parts] = doc_path.as_posix()
    
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"# {identifier}\n\n")
        fd.write(f"::: {identifier}\n")
        fd.write("    options:\n")
        fd.write("      show_signature_annotations: true\n")
        fd.write("      show_source: true\n")
        fd.write("      show_root_heading: true\n")
        fd.write("      show_category_heading: true\n")
        fd.write("      docstring_style: numpy\n")
        fd.write("      merge_init_into_class: true\n")
    
    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Write the navigation file
with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

print("API documentation generated successfully!")
