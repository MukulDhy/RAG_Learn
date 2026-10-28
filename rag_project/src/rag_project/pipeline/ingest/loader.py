from pathlib import Path

PDF_DIR = "ai_ml_knowledge_base"

def main() -> None:
    
    folder = Path(PDF_DIR)
    output_file = Path("pdf_paths.txt")

    pdf_paths = sorted(folder.rglob("*"))

    with output_file.open("w", encoding="utf-8") as f:
        for path in pdf_paths:
            if path.is_file() and path.suffix.lower() == ".pdf":
                f.write(str(path.resolve()) + "\n")

    print(f"PDF paths saved to {output_file.resolve()}")


if __name__ == "__main__":
    main()