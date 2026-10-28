"""Create a deployment ZIP from an explicit allowlist, including local weights."""
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    files = ["app.py", "start_cloud.py", "requirements.txt", "requirements-cloud.txt",
             "Dockerfile", ".dockerignore", "render.yaml", ".env.example", "README.md",
             "LAB3.md", "best.pt"]
    sources = [root / name for name in files]
    sources += [path for folder in ("templates", "static") for path in (root / folder).rglob("*") if path.is_file()]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = root / "outputs" / "masklens-lab3-cloud.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(root))
    print(f"Created {output} ({output.stat().st_size:,} bytes). No .env or detection logs included.")


if __name__ == "__main__":
    main()
