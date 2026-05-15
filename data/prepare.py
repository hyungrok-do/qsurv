"""Download and convert the public tabular survival datasets to data/files/<name>.csv.

Run once before training:
    python data/prepare.py

NWTCO must be obtained separately (R's `survival` package) and saved to data/files/nwtco.csv;
Framingham requires application access and saved to data/files/framingham.csv.
"""
import io
import os
import zipfile

import pandas as pd
import requests
from pycox.datasets import metabric
from sksurv.datasets import load_flchain, load_gbsg2


OUT_DIR = 'data/files'


def _process_sksurv(X, y, name):
    """sksurv returns y as a structured array; flatten it to event/duration columns and join with X."""
    event_field, time_field = y.dtype.names[0], y.dtype.names[1]
    y_df = pd.DataFrame(y).rename(columns={event_field: 'event', time_field: 'duration'})
    y_df['event'] = y_df['event'].astype(int)
    df = pd.concat([X, y_df], axis=1)
    df.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
    print(f"Saved {OUT_DIR}/{name}.csv")
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Processing Metabric...")
    metabric.read_df().to_csv(os.path.join(OUT_DIR, 'metabric.csv'), index=False)
    print(f"Saved {OUT_DIR}/metabric.csv")

    print("Processing Support2...")
    r = requests.get("https://hbiostat.org/data/repo/support2csv.zip")
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open('support2.csv') as f:
            pd.read_csv(f).to_csv(os.path.join(OUT_DIR, 'support2.csv'), index=False)
    print(f"Saved {OUT_DIR}/support2.csv")

    print("Processing FLChain...")
    X_fl, y_fl = load_flchain()
    # 'chapter' encodes the cause of death (target leak); the two `sample.yr`/`flc.grp`
    # are administrative and not used as features.
    X_fl = X_fl.drop(['sample.yr', 'flc.grp', 'chapter'], axis=1)
    _process_sksurv(X_fl, y_fl, 'flchain')

    print("Processing GBSG2...")
    X_g, y_g = load_gbsg2()
    _process_sksurv(X_g, y_g, 'gbsg2')


if __name__ == '__main__':
    main()
