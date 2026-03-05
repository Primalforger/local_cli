"""Archive / compression tools — create, extract, list archives."""

import shutil
from pathlib import Path
from tools.common import console, _sanitize_tool_args, _sanitize_path_arg, _validate_path, _confirm


def tool_archive_create(args: str) -> str:
    """Create an archive (zip, tar.gz, tar.bz2)."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    if len(parts) != 2:
        return "Error: Use format output_path|source_dir"

    output_path = _sanitize_path_arg(parts[0])
    source_dir = _sanitize_path_arg(parts[1])

    src, error = _validate_path(source_dir)
    if error:
        return error

    out = Path(output_path).resolve()
    try:
        out.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot write outside project directory: {output_path}"

    console.print(f"\n[yellow]Create archive:[/yellow] {output_path} from {source_dir}")
    if not _confirm("Proceed? (y/n): "):
        return "Cancelled."

    try:
        if output_path.endswith(".zip"):
            actual_path = shutil.make_archive(
                str(out.with_suffix('')), 'zip',
                root_dir=str(src.parent), base_dir=src.name,
            )
        elif output_path.endswith(".tar.gz") or output_path.endswith(".tgz"):
            actual_path = shutil.make_archive(
                str(out).replace('.tar.gz', '').replace('.tgz', ''), 'gztar',
                root_dir=str(src.parent), base_dir=src.name,
            )
        elif output_path.endswith(".tar.bz2"):
            actual_path = shutil.make_archive(
                str(out).replace('.tar.bz2', ''), 'bztar',
                root_dir=str(src.parent), base_dir=src.name,
            )
        elif output_path.endswith(".tar"):
            actual_path = shutil.make_archive(
                str(out.with_suffix('')), 'tar',
                root_dir=str(src.parent), base_dir=src.name,
            )
        else:
            return "Error: Unsupported format. Use .zip, .tar.gz, .tar.bz2, or .tar"

        size = Path(actual_path).stat().st_size
        return f"\u2713 Created archive: {output_path} ({size:,} bytes)"
    except Exception as e:
        return f"Error creating archive: {e}"


def tool_archive_extract(args: str) -> str:
    """Extract an archive."""
    cleaned = _sanitize_tool_args(args)
    parts = cleaned.split("|")
    archive_path = _sanitize_path_arg(parts[0])
    dest_dir = _sanitize_path_arg(parts[1]) if len(parts) > 1 else "."

    path, error = _validate_path(archive_path)
    if error:
        return error

    dest = Path(dest_dir).resolve()
    try:
        dest.relative_to(Path.cwd().resolve())
    except ValueError:
        return f"Error: Cannot extract outside project directory: {dest_dir}"

    console.print(f"\n[yellow]Extract:[/yellow] {archive_path} \u2192 {dest_dir}")
    if not _confirm("Proceed? (y/n): "):
        return "Cancelled."

    try:
        shutil.unpack_archive(str(path), str(dest))
        return f"\u2713 Extracted {archive_path} \u2192 {dest_dir}"
    except Exception as e:
        return f"Error extracting archive: {e}"


def tool_archive_list(args: str) -> str:
    """List contents of an archive."""
    filepath = _sanitize_path_arg(args)
    path, error = _validate_path(filepath)
    if error:
        return error

    try:
        import zipfile
        import tarfile

        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                entries = zf.namelist()
                output = f"Archive: {filepath} (ZIP, {len(entries)} entries)\n"
                for entry in entries[:100]:
                    info = zf.getinfo(entry)
                    output += f"  {entry} ({info.file_size:,} bytes)\n"
                if len(entries) > 100:
                    output += f"  ... and {len(entries) - 100} more\n"
                return output

        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                members = tf.getmembers()
                output = f"Archive: {filepath} (TAR, {len(members)} entries)\n"
                for m in members[:100]:
                    output += f"  {m.name} ({m.size:,} bytes)\n"
                if len(members) > 100:
                    output += f"  ... and {len(members) - 100} more\n"
                return output

        return f"Error: Not a recognized archive format: {filepath}"
    except Exception as e:
        return f"Error reading archive: {e}"
