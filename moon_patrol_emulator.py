"""
Shortcut kept for existing run configurations: same as `python mame.py mpatrol`.
"""

import sys
import mame

if __name__ == "__main__":
    sys.exit(mame.main(["mpatrol"] + sys.argv[1:]))
