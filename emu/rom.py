"""
ROM set loading (the job of MAME's romload.cpp, much simplified: no CRC check yet).
"""

import os
from typing import Dict, Iterable, List, Optional


class RomSet:
    """The files of one game, looked up in roms/<set name>/ (first directory that exists)."""

    def __init__(self, roms_root: str, set_names: Iterable[str]):
        self.directory: Optional[str] = None
        for name in set_names:
            path = os.path.join(roms_root, name)
            if os.path.isdir(path):
                self.directory = path
                break
        self.files: Dict[str, bytes] = {}
        self.missing: List[str] = []

    def load(self, filenames: Iterable[str]) -> bool:
        """Read every file; returns False (and fills self.missing) if any is absent."""
        for filename in filenames:
            path = os.path.join(self.directory, filename) if self.directory else None
            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    self.files[filename] = f.read()
            else:
                self.missing.append(filename)
        return not self.missing

    def __getitem__(self, filename: str) -> bytes:
        return self.files[filename]
