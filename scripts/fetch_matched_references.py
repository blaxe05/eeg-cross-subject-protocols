"""Fetch pinned upstream model software into an external reference directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

REFERENCES={
    'eegnet':('vlawhern/arl-eegmodels','4a512e503198db2010848813ead9afbf8cd54c97'),
    'tsception':('yi-ding-cs/TSception','9efd666b618d006e32e6da1d30dbc79b1d190604'),
    'rgnn':('zhongpeixiang/RGNN','685b1a2645185bd8129c04a789dbfde8ad896a59'),
    'nsal_dgat':('YYingDL/NSAL-DGAT','a55dfca74caaa5095a89c926c5d0af35e3ed4b18'),
    'ms_mda':('VoiceBeer/MS-MDA','a2946c38dc6b8d75e9a81bc1aa3651b1cb70f240'),
    'libeer':('XJTU-EEG/LibEER','dddff9776dbdae21195fe320dff0a5ba61628a18')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    a.destination.mkdir(parents=True,exist_ok=True);records={}
    for name,(repository,commit) in REFERENCES.items():
        folder=a.destination/name
        if not folder.exists():
            subprocess.run(['git','clone','--no-checkout','https://github.com/'+repository+'.git',str(folder)],check=True)
            subprocess.run(['git','-C',str(folder),'checkout',commit],check=True)
        actual=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        assert actual==commit,f'{name}: existing reference differs from the pinned revision'
        assert not subprocess.check_output(['git','-C',str(folder),'status','--porcelain'],text=True).strip()
        records[name]={'repository':'https://github.com/'+repository,'commit':commit}
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
