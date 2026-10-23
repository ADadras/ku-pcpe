"""Linux Landlock test harness: deny filesystem access outside explicit roots and TCP."""

import argparse, ctypes, os, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--read", action="append", default=[])
p.add_argument("--write", action="append", default=[])
p.add_argument("--must-deny", action="append", default=[])
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
lib = ctypes.CDLL(None, use_errno=True)
abi = lib.syscall(444, 0, 0, 1)
if abi < 4:
    raise RuntimeError("This isolation check requires Landlock ABI 4+")


class Ruleset(ctypes.Structure):
    _fields_ = [("fs", ctypes.c_uint64), ("net", ctypes.c_uint64)]


class PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("access", ctypes.c_uint64), ("parent", ctypes.c_int32)]


rights = (1 << 15) - 1
attr = Ruleset(rights, 3)
fd = lib.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
if fd < 0:
    raise OSError(ctypes.get_errno(), "create ruleset")
for path, access in [(p, 13) for p in a.read] + [(p, rights) for p in a.write]:
    opened = os.open(Path(path).resolve(), os.O_PATH | os.O_CLOEXEC)
    rule = PathRule(access, opened)
    if lib.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
        raise OSError(ctypes.get_errno(), "add path " + path)
    os.close(opened)
if lib.prctl(38, 1, 0, 0, 0) < 0 or lib.syscall(446, fd, 0) < 0:
    raise OSError(ctypes.get_errno(), "restrict self")
os.close(fd)
import socket, errno

probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
code = probe.connect_ex(("127.0.0.1", 9))
probe.close()
if code not in [errno.EACCES, errno.EPERM]:
    raise RuntimeError("TCP isolation failed")
print("TCP denied as required", flush=True)
for path in a.must_deny:
    try:
        open(path, "rb")
    except PermissionError:
        print("DENIED as required: " + Path(path).name, flush=True)
    else:
        raise RuntimeError("Isolation failure: an excluded source was readable")
command = a.command[1:] if a.command and a.command[0] == "--" else a.command
if not command:
    raise ValueError("Missing command")
os.execvp(command[0], command)
