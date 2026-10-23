"""Atomic task checkpoints keyed by immutable inputs and evaluator source."""

from pathlib import Path
import hashlib, json, os


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TaskCache:
    def __init__(self, root, signature, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=resume)
        manifest = self.root / "signature.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != signature:
                raise ValueError("Resume signature differs from inputs or code")
        else:
            manifest.write_text(json.dumps(signature, sort_keys=True))

    def path(self, task):
        return self.root / (hashlib.sha256(task.encode()).hexdigest() + ".json")

    def load(self, task):
        path = self.path(task)
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        assert row["task_id"] == task
        encoded = json.dumps(
            row["result"], sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != row["sha256"]:
            raise ValueError("Checkpoint content is corrupt")
        return row["result"]

    def save(self, task, result):
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        row = dict(
            task_id=task, result=result, sha256=hashlib.sha256(encoded).hexdigest()
        )
        path = self.path(task)
        temporary = path.with_suffix("." + str(os.getpid()) + ".tmp")
        with temporary.open("w") as f:
            json.dump(row, f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
