import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)
