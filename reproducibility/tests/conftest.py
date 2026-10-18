from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'generated/.mplconfig'))
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT))
