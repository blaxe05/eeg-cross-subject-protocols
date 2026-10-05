"""Correct only evaluation dropout; preserve training and checkpoint parameter names."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/p3_references/LibEER/LibEER'))
from models.CDCN import CDCN as HistoricalCDCN, Convblock

class CorrectedConvblock(Convblock):
    def forward(self,x):
        output=self.conv(self.pad(self.relu(self.bn(x))))
        output=F.dropout(output,p=.5,training=self.training)
        return torch.cat([x,output],1)

class CorrectedCDCN(HistoricalCDCN):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for module in self.modules():
            if type(module) is Convblock:
                module.__class__=CorrectedConvblock

    def forward(self,x):
        x=x.unsqueeze(1)
        x=self.features(x)
        x=self.block_tran(x)
        x=self.trail(x)
        x=self.GAP(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        return self.fc(x)

def make_model(name,root):
    if name=='CDCN':return CorrectedCDCN(62,5,3,dropout=.5)
    from src.p3_protocol_benchmark.models import make_model as historical
    return historical(name,root)

from src.p3_protocol_benchmark.models import logits
