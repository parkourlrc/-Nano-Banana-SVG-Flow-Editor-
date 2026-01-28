import os
import shutil
import subprocess
import sys


def _run(cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(cmd)
    print(f"+ {printable}", flush=True)
    subprocess.check_call(cmd, cwd=cwd, env=env)


def main(argv: list[str]) -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, ".weights")
    os.makedirs(weights_dir, exist_ok=True)

    repo_url = os.environ.get("SAM3_MODELSCOPE_REPO", "https://www.modelscope.cn/models/facebook/sam3.git")
    ckpt_name = os.environ.get("SAM3_CHECKPOINT_NAME", "sam3.pt")
    if len(argv) >= 2 and argv[1] and not argv[1].startswith("-"):
        ckpt_name = str(argv[1]).strip()

    dest_path = os.path.join(weights_dir, ckpt_name)
    repo_dir = os.path.join(weights_dir, ".sam3_modelscope_repo")

    if shutil.which("git") is None:
        print("ERROR: `git` not found in PATH.", file=sys.stderr)
        return 2
    try:
        _run(["git", "lfs", "version"])
    except Exception:
        print("ERROR: `git lfs` not available. Please install Git LFS and retry.", file=sys.stderr)
        return 2

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 50_000_000:
        print(f"OK: checkpoint already exists: {dest_path}")
        return 0

    env_clone = os.environ.copy()
    env_clone.setdefault("GIT_LFS_SKIP_SMUDGE", "1")  # avoid downloading huge weights during clone

    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        _run(["git", "clone", "--depth", "1", repo_url, repo_dir], env=env_clone)

    _run(["git", "lfs", "pull", "--include", ckpt_name], cwd=repo_dir)

    src_path = os.path.join(repo_dir, ckpt_name)
    if not os.path.exists(src_path):
        print(f"ERROR: checkpoint file not found after pull: {src_path}", file=sys.stderr)
        return 2
    size = int(os.path.getsize(src_path))
    if size <= 50_000_000:
        head = ""
        try:
            with open(src_path, "rb") as f:
                head = f.read(256).decode("utf-8", errors="replace")
        except Exception:
            head = ""
        print(
            f"ERROR: checkpoint looks too small (size={size} bytes). It may still be a Git LFS pointer.\n{head}",
            file=sys.stderr,
        )
        return 2

    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except Exception:
            pass

    try:
        os.link(src_path, dest_path)  # hardlink (no extra disk)
        print(f"OK: created hardlink: {dest_path}")
    except Exception as exc:
        print(f"[warn] hardlink failed ({exc}); copying instead...", file=sys.stderr)
        shutil.copy2(src_path, dest_path)
        print(f"OK: copied checkpoint: {dest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

