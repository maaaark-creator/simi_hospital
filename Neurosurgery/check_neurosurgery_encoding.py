from __future__ import annotations

from pathlib import Path


FILES_TO_CHECK = [
    "neurosurgery_patient_mapper_agent.py",
    "neurosurgery_decision_agent.py",
    "neurosurgery_kb_retriever.py",
    "run_neurosurgery_pipeline_demo.py",
    "neurosurgery_kb.json",
]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    for name in FILES_TO_CHECK:
        path = base_dir / name
        text = path.read_text(encoding="utf-8")
        preview = next((line.strip() for line in text.splitlines() if line.strip()), "")
        print(f"{name}: utf-8 OK | preview: {preview[:80]}")


if __name__ == "__main__":
    main()
