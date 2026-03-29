import platform
import socket
import psutil
from datetime import datetime

import git
import h5py

def get_git_info() -> dict:
    """Get current git hash and working tree state using GitPython."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return {
            "commit_hash": repo.head.commit.hexsha,
            "branch":      repo.active_branch.name,
            "is_dirty":    str(repo.is_dirty()),          # staged or unstaged changes
            "diff_stat":   repo.git.diff("--stat") or "clean",
        }
    except git.InvalidGitRepositoryError:
        return {"error": "not a git repository"}
    except git.GitCommandError as e:
        return {"error": str(e)}

def get_system_info() -> dict:
    """Get basic information about the current machine."""
    mem = psutil.virtual_memory()
    return {
        "hostname":       socket.gethostname(),
        "os":             platform.platform(),
        "cpu":            platform.processor(),
        "cpu_cores":      str(psutil.cpu_count(logical=False)),
        "cpu_threads":    str(psutil.cpu_count(logical=True)),
        "ram_total_gb":   f"{mem.total / 1e9:.2f}",
        "python_version": platform.python_version(),
        "timestamp":      datetime.utcnow().isoformat() + "Z",
    }

def save_metadata(h5file: h5py.File):
    """Write a /metadata group with git + system info into an open HDF5 file."""
    meta = h5file.require_group("metadata")

    git_grp = meta.require_group("git")
    for k, v in get_git_info().items():
        git_grp.attrs[k] = v

    sys_grp = meta.require_group("system")
    for k, v in get_system_info().items():
        sys_grp.attrs[k] = v