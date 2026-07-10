import argparse
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument('--results-root',required=True); a=p.parse_args(); root=Path(a.results_root)
 dfs=[pd.read_csv(p) for p in root.glob('*/*metrics.csv') if p.name=='metrics.csv']
 out=root/'all_metrics.csv'
 if dfs: pd.concat(dfs,ignore_index=True).to_csv(out,index=False)
 print(out)
if __name__=='__main__': main()
