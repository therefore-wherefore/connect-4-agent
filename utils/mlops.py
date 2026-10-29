import subprocess
import hashlib

def get_git_commit_hash():
    """Extracts current Git commit hash. Fails strictly if git is unavailable or invalid."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("ascii").strip()

def check_git_cleanliness():
    """Enforces a clean Git working tree. Raises an error if uncommitted changes exist."""
    status_output = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.STDOUT).decode("ascii").strip()
    if status_output:
        raise RuntimeError(
            "MANDATORY MLOPS CHECK FAILED: Uncommitted changes detected in the Git working tree.\n"
            "You must commit all changes before training to guarantee absolute model reproducibility."
        )

def get_data_checksum(X, y):
    """Computes an MD5 checksum of the training arrays to fingerprint the dataset."""
    hasher = hashlib.md5()
    hasher.update(X.tobytes())
    hasher.update(y.tobytes())
    return hasher.hexdigest()