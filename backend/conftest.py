import sys
from pathlib import Path

# Add project root and backend dir to sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = Path(__file__).resolve().parent

if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
