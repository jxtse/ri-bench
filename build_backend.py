"""Small PEP 517 wheel backend: no installation of build dependencies needed."""

import base64
import csv
import hashlib
import io
from pathlib import Path
import tarfile
import tomllib
import zipfile


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None, *, _editable=False):
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    normalized = project["name"].replace("-", "_")
    version = project["version"]
    name = f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {project['name']}\nVersion: {version}\nRequires-Python: {project['requires-python']}\n"
    metadata += "".join(f"Requires-Dist: {dependency}\n" for dependency in project["dependencies"])
    for extra, dependencies in project.get("optional-dependencies", {}).items():
        metadata += f"Provides-Extra: {extra}\n"
        metadata += "".join(f'Requires-Dist: {dependency}; extra == "{extra}"\n' for dependency in dependencies)
    if _editable:
        files = {normalized + ".pth": (str(Path("src").resolve()) + "\n").encode()}
    else:
        files = {str(path.relative_to("src")): path.read_bytes()
                 for package in ("ribench_agentic",) for path in Path("src", package).rglob("*")
                 if path.is_file() and path.suffix in {".py", ".json"}}
    files[dist_info + "/METADATA"] = metadata.encode()
    files[dist_info + "/WHEEL"] = b"Wheel-Version: 1.0\nGenerator: ribench-local-backend\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[dist_info + "/entry_points.txt"] = ("[console_scripts]\n" + "\n".join(f"{key} = {value}" for key, value in project["scripts"].items()) + "\n").encode()
    record = io.StringIO()
    writer = csv.writer(record)
    for path, contents in sorted(files.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).decode().rstrip("=")
        writer.writerow([path, "sha256=" + encoded, len(contents)])
    writer.writerow([dist_info + "/RECORD", "", ""])
    files[dist_info + "/RECORD"] = record.getvalue().encode()
    Path(wheel_directory).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(wheel_directory) / name, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, contents in sorted(files.items()):
            entry = zipfile.ZipInfo(path, date_time=(2026, 9, 6, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, contents)
    return name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return build_wheel(wheel_directory, config_settings, metadata_directory, _editable=True)


def build_sdist(sdist_directory, config_settings=None):
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    stem = f"{project['name'].replace('-', '_')}-{project['version']}"
    name = f"{stem}.tar.gz"
    Path(sdist_directory).mkdir(parents=True, exist_ok=True)
    paths = [Path(value) for value in ("pyproject.toml", "build_backend.py", "README.md") if Path(value).exists()]
    paths += [path for folder in ("src", "tests") for path in Path(folder).rglob("*") if path.is_file() and path.suffix in {".py", ".json", ".jsonl"}]
    with tarfile.open(Path(sdist_directory) / name, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=f"{stem}/" + str(path))
    return name
