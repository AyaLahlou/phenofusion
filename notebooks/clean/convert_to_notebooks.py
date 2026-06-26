"""
Convert Python Scripts to Jupyter Notebooks
============================================

This script converts the refactored Python scripts to Jupyter notebooks
while preserving structure and documentation.

Usage:
    python convert_to_notebooks.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict


def python_to_cells(python_file: str) -> List[Dict]:
    """
    Convert Python file to Jupyter notebook cells

    Args:
        python_file: Path to Python file

    Returns:
        List of cell dictionaries
    """
    with open(python_file, "r") as f:
        content = f.read()

    cells = []

    # Split by major sections (marked by ===)
    sections = re.split(r"# ={50,}\n# (.+?)\n# ={50,}", content)

    # Process sections
    i = 0
    while i < len(sections):
        section_content = sections[i].strip()

        if i + 1 < len(sections):
            # This is a section title
            section_title = sections[i + 1]
            section_code = sections[i + 2] if i + 2 < len(sections) else ""

            # Create markdown cell for title
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"## {section_title}\n"],
                }
            )

            # Process code in this section
            if section_code.strip():
                # Split by classes/functions
                parts = re.split(
                    r"(^class |^def |^if __name__)", section_code, flags=re.MULTILINE
                )

                current_code = []
                for part in parts:
                    if part.strip():
                        if part in ["class ", "def ", "if __name__"]:
                            # Start of new definition
                            if current_code:
                                # Save previous code block
                                cells.append(
                                    {
                                        "cell_type": "code",
                                        "execution_count": None,
                                        "metadata": {},
                                        "outputs": [],
                                        "source": ["".join(current_code)],
                                    }
                                )
                                current_code = []
                            current_code.append(part)
                        else:
                            current_code.append(part)

                # Add remaining code
                if current_code:
                    cells.append(
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": ["".join(current_code)],
                        }
                    )

            i += 3
        else:
            # Process remaining content
            if section_content:
                # Check if it's a docstring
                if section_content.startswith('"""') or section_content.startswith(
                    "'''"
                ):
                    # Extract docstring as markdown
                    docstring = re.search(
                        r'["\']{{3}}(.+?)["\']{{3}}', section_content, re.DOTALL
                    )
                    if docstring:
                        cells.append(
                            {
                                "cell_type": "markdown",
                                "metadata": {},
                                "source": [docstring.group(1).strip() + "\n"],
                            }
                        )
                        # Remaining as code
                        remaining = section_content[docstring.end() :].strip()
                        if remaining:
                            cells.append(
                                {
                                    "cell_type": "code",
                                    "execution_count": None,
                                    "metadata": {},
                                    "outputs": [],
                                    "source": [remaining + "\n"],
                                }
                            )
                else:
                    cells.append(
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [section_content + "\n"],
                        }
                    )
            i += 1

    return cells


def create_notebook(python_file: str, output_file: str):
    """
    Create Jupyter notebook from Python file

    Args:
        python_file: Input Python file
        output_file: Output notebook file
    """
    print(f"Converting {python_file} to {output_file}...")

    # Read Python file
    with open(python_file, "r") as f:
        code = f.read()

    # Simple conversion: create cells for each major section
    cells = []

    # Add title cell
    title = Path(python_file).stem.replace("_", " ").title()
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n\nAutomatically converted from Python script.\n"],
        }
    )

    # Split by class/function definitions
    current_chunk = []

    for line in code.split("\n"):
        # Check for class or function definition
        if (
            line.startswith("class ")
            or line.startswith("def ")
            or line.startswith("if __name__")
        ):
            # Save previous chunk
            if current_chunk:
                chunk_text = "\n".join(current_chunk)
                if chunk_text.strip():
                    cells.append(
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": [chunk_text + "\n"],
                        }
                    )
                current_chunk = []

        current_chunk.append(line)

    # Add remaining chunk
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        if chunk_text.strip():
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [chunk_text + "\n"],
                }
            )

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Write notebook
    with open(output_file, "w") as f:
        json.dump(notebook, f, indent=2)

    print(f"✓ Created {output_file}")


def main():
    """Main conversion function"""
    print("=" * 80)
    print("CONVERTING PYTHON SCRIPTS TO JUPYTER NOTEBOOKS")
    print("=" * 80)

    # Files to convert
    files = [
        ("sos_eos_detection_refactored.py", "sos_eos_detection_refactored.ipynb"),
        (
            "feature_importance_extraction_refactored.py",
            "feature_importance_extraction_refactored.ipynb",
        ),
    ]

    output_dir = Path(".")

    for python_file, notebook_file in files:
        if Path(python_file).exists():
            output_path = output_dir / notebook_file
            create_notebook(python_file, str(output_path))
        else:
            print(f"⚠️  Warning: {python_file} not found, skipping...")

    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print("\nYou can now open the notebooks in Jupyter:")
    print("  jupyter notebook sos_eos_detection_refactored.ipynb")
    print("  jupyter notebook feature_importance_extraction_refactored.ipynb")
    print("\nOr use JupyterLab:")
    print("  jupyter lab")


if __name__ == "__main__":
    main()
