"""Model interfaces for common-input, subject-disjoint EEG comparisons.

Author repositories are supplied separately. No third-party source code or
licensed EEG metadata is redistributed by this module.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_reference(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hemisphere_order(channels):
    """Retain bilateral electrode pairs; left pairs precede right pairs."""
    names = [str(c).upper() for c in channels]
    pairs = []
    seen = set()
    for name in names:
        prefix = name.rstrip('0123456789')
        suffix = name[len(prefix):]
        if not suffix:
            continue
        number = int(suffix)
        left = prefix + str(number if number % 2 else number-1)
        right = prefix + str(number+1 if number % 2 else number)
        if left in names and right in names and left not in seen:
            pairs.append((names.index(left), names.index(right)))
            seen.add(left)
    if not pairs:
        raise ValueError('TSception requires identifiable bilateral electrode pairs')
    return [p[0] for p in pairs] + [p[1] for p in pairs]


class EEGNet(nn.Module):
    """EEGNet-8,2 with grouped depthwise filters and post-update max norms.

    Layer order follows vlawhern/arl-eegmodels EEGNet. Torch BatchNorm
    momentum is 1 minus Keras momentum. Temporal kernels span 0.5 seconds.
    """
    def __init__(self, channels, samples, classes, sampling_rate):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, sampling_rate//2), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=.001, momentum=.01)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16, eps=.001, momentum=.01)
        self.depthwise = nn.Conv2d(16, 16, (1, 16), padding='same', groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16, eps=.001, momentum=.01)
        self.classifier = nn.Linear(16*(samples//32), classes)
        self.drop = nn.Dropout(.5)
        for layer in [self.temporal, self.spatial, self.depthwise, self.pointwise, self.classifier]:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.classifier.bias)

    @torch.no_grad()
    def constrain(self):
        for parameter, limit in [(self.spatial.weight, 1.), (self.classifier.weight, .25)]:
            norms = parameter.flatten(1).norm(dim=1).clamp_min(1e-12)
            factor = torch.clamp(limit/norms, max=1.)
            parameter.mul_(factor.view(-1, *([1]*(parameter.ndim-1))))

    def forward(self, x):
        x = self.bn1(self.temporal(x.unsqueeze(1)))
        x = self.drop(F.avg_pool2d(F.elu(self.bn2(self.spatial(x))), (1, 4)))
        x = self.drop(F.avg_pool2d(F.elu(self.bn3(self.pointwise(self.depthwise(x)))), (1, 8)))
        return self.classifier(x.permute(0,2,3,1).flatten(1))


class TSception(nn.Module):
    def __init__(self, references, channels, samples, classes, sampling_rate):
        super().__init__()
        module = load_reference(Path(references)/'tsception/code/networks.py', 'matched_tsception_reference')
        self.register_buffer('order', torch.tensor(hemisphere_order(channels), dtype=torch.long))
        self.network = module.TSception(classes, (1, len(self.order), samples), sampling_rate, 15, 15, 32, .5)

    def forward(self, x):
        return self.network(x.index_select(1, self.order).unsqueeze(1))


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient*gradient, None


def anatomical_prior(channels):
    """Inverse-square distances with differential-asymmetry homologous edges."""
    names = [str(c).upper() for c in channels]
    if len(names)==62:
        # SEED includes CB1/CB2, absent from the standard 10-20 montage.
        # Use the pinned reference prior in its explicit electrode order.
        from data_utils.constants.seed import SEED_CHANNEL_NAME, SEED_RGNN_ADJACENCY_MATRIX
        original=[str(c).upper() for c in SEED_CHANNEL_NAME]
        assert set(original)==set(names)
        order=[original.index(c) for c in names]
        return torch.tensor(np.asarray(SEED_RGNN_ADJACENCY_MATRIX)[np.ix_(order,order)],dtype=torch.float32)
    import mne
    positions = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    lookup = {name.upper(): value for name, value in positions.items()}
    coordinates = np.array([lookup[str(c).upper()] for c in channels])*100.
    distance = ((coordinates[:, None]-coordinates[None, :])**2).sum(-1)
    prior = np.minimum(5./np.maximum(distance, 1e-12), 1.)
    np.fill_diagonal(prior, 1.)
    # The author prior uses nine long-distance homologous pairs on SEED.
    pairs = [('FP1','FP2'),('AF3','AF4'),('F7','F8'),('FT7','FT8'),
             ('T7','T8'),('TP7','TP8'),('P7','P8'),('PO7','PO8'),('O1','O2')]
    for left, right in pairs:
        if left in names and right in names:
            a, b = names.index(left), names.index(right)
            prior[a, b] -= 1.; prior[b, a] -= 1.
    return torch.tensor(prior, dtype=torch.float32)


class RGNN(nn.Module):
    """Dense equivalent of the author's signed, two-hop simplified graph layer."""
    def __init__(self, channels, classes, prior, hidden=64):
        super().__init__()
        index = torch.tril_indices(channels, channels)
        self.register_buffer('tri_index', index)
        self.edges = nn.Parameter(prior[index[0], index[1]].clone())
        self.projection = nn.Linear(5, hidden)
        self.classifier = nn.Linear(hidden, classes)
        self.domain = nn.Linear(hidden, 2)
        self.channels, self.classes = channels, classes

    def adjacency(self):
        matrix = self.edges.new_zeros((self.channels, self.channels))
        matrix[self.tri_index[0], self.tri_index[1]] = self.edges
        matrix = matrix + matrix.T - torch.diag(matrix.diagonal())
        degree = matrix.abs().sum(1).clamp_min(1e-12).rsqrt()
        return degree[:, None]*matrix*degree[None, :]

    def features(self, x):
        adjacency = self.adjacency()
        return F.relu(self.projection(adjacency @ (adjacency @ x)))

    def forward(self, x):
        features = self.features(x).sum(1)
        return self.classifier(F.dropout(features, p=.7, training=self.training))

    def classification_loss(self, logits, labels):
        if self.classes == 3:
            targets = logits.new_tensor([[1-2*.1/3, 2*.1/3, 0], [.1/3, 1-2*.1/3, .1/3], [0, 2*.1/3, 1-2*.1/3]])[labels]
        elif self.classes == 4:
            targets = logits.new_tensor([[.925,.025,.025,.025],[.1/3,1-2*.1/3,.1/3,0],
                                         [.025,.025,.925,.025],[.1/3,0,.1/3,1-2*.1/3]])[labels]
        elif self.classes == 2:
            targets = F.one_hot(labels, 2).to(logits.dtype)
        else:
            # The source reference defines no nine-class emotion-distance matrix.
            targets = F.one_hot(labels, self.classes).to(logits.dtype)*.9 + .1/self.classes
        return -(targets*F.log_softmax(logits, 1)).sum(1).mean()

    def training_loss(self, x, labels, auxiliary, progress):
        features = self.features(x)
        logits = self.classifier(F.dropout(features.sum(1), p=.7, training=True))
        loss = self.classification_loss(logits, labels) + .01*self.edges.abs().sum()
        if auxiliary is not None:
            joined = torch.cat([features, self.features(auxiliary)])
            coefficient = 2/(1+math.exp(-10*progress))-1
            logits_domain = self.domain(ReverseGradient.apply(joined, coefficient)).flatten(0, 1)
            domains = torch.cat([torch.zeros(len(x)*self.channels, dtype=torch.long, device=x.device),
                                 torch.ones(len(auxiliary)*self.channels, dtype=torch.long, device=x.device)])
            loss = loss + F.cross_entropy(logits_domain, domains)
        return loss


class MSMDA(nn.Module):
    """Multi-source marginal adaptation, retaining all domain-specific heads."""
    def __init__(self, channels, classes, sources):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(channels*5,256),nn.LeakyReLU(.01),
                                    nn.Linear(256,128),nn.LeakyReLU(.01),nn.Linear(128,64),nn.LeakyReLU(.01))
        self.extractors, self.classifiers = nn.ModuleList(), nn.ModuleList()
        for _ in range(sources):
            self.extractors.append(nn.Sequential(nn.Linear(64,32),nn.BatchNorm1d(32),nn.LeakyReLU(.01)))
            self.classifiers.append(nn.Linear(32,classes))

    def forward(self, x):
        shared = self.shared(x.flatten(1))
        probability = torch.stack([F.softmax(head(extractor(shared)),1)
                                   for extractor,head in zip(self.extractors,self.classifiers)]).mean(0)
        return probability.clamp_min(1e-12).log()

    def training_loss(self, x, labels, auxiliary, mark, progress):
        source = self.shared(x.flatten(1))
        target = self.shared(auxiliary.flatten(1))
        target_features = [extractor(target) for extractor in self.extractors]
        source_features = self.extractors[mark](source)
        classification = F.cross_entropy(self.classifiers[mark](source_features), labels)
        delta = source_features-target_features[mark]
        mmd = delta.mean(0).square().sum()
        discrepancy = sum((F.softmax(target_features[mark],1)-F.softmax(value,1)).abs().mean()
                          for i,value in enumerate(target_features) if i != mark)
        coefficient = 2/(1+math.exp(-10*progress))-1
        return classification + coefficient*mmd + coefficient/100*discrepancy


class NSALDGAT(nn.Module):
    """Author NSAL-DGAT network with explicit, resumable source-only banks."""
    def __init__(self, references, channels, classes, source_count, device):
        super().__init__()
        path = Path(references)/'nsal_dgat'
        source = load_reference(path/'network.py', 'matched_nsal_network')
        adversarial = load_reference(path/'Adversarial.py', 'matched_nsal_adversarial')
        self.network = source.Domain_adaption_model([5,channels],2,256,64,classes,device,source_count)
        class GlobalMaxPool(nn.Module):
            def forward(this,x):return x.flatten(2).max(dim=2).values[:,:,None,None]
        for module in self.network.modules():
            if hasattr(module,'maxpool') and isinstance(module.maxpool,nn.AdaptiveMaxPool2d):
                assert module.maxpool.output_size in [1,(1,1)]
                module.maxpool=GlobalMaxPool()
        # The demo hard-codes reshape(5, 62). Preserve band/electrode axes and
        # parameterize only the dataset dimensions already accepted by __init__.
        def encoder_forward(this,x):
            x=x.transpose(1,2).unsqueeze(2)
            features,adjacency=this.GGCN(x)
            features,ca,sa=this.CBAM(features)
            value=this.dropout1(F.relu(this.fc1(features.flatten(1))))
            value=this.dropout2(F.relu(this.fc2(value)))
            return value,[adjacency,ca,sa]
        self.network.encoder.forward=types.MethodType(encoder_forward,self.network.encoder)
        for module in self.network.modules():
            if hasattr(module, 'A') and isinstance(module.A, torch.Tensor):
                value = module.A; delattr(module, 'A'); module.register_buffer('A',value)
        # Legacy banks are plain attributes and would otherwise escape checkpoints.
        for name in ['source_f_bank','source_score_bank']:
            value = getattr(self.network,name); delattr(self.network,name)
            self.network.register_buffer(name,value.to(device),persistent=False)
        discriminator=source.Discriminator(64)
        # DAANLoss uses BCEWithLogitsLoss, so the discriminator must emit logits.
        discriminator.sigmoid=nn.Identity()
        self.domain_loss = adversarial.DAANLoss(discriminator,num_class=classes,max_iter=1000)
        # A zero-degree column sends no messages. Avoid reciprocal(0) without
        # changing the reference calculation on nonisolated columns.
        for module in self.network.modules():
            if isinstance(module,source.HGCN):
                def guarded_forward(this,x,matrix):
                    degree=matrix.sum(0)
                    inverse=torch.where(degree>0,degree.clamp_min(1e-12).reciprocal(),torch.zeros_like(degree))
                    operator=matrix*inverse[None,:]
                    return this.resGCN(x,x,operator).contiguous()
                module.forward=types.MethodType(guarded_forward,module)
        # Bank updates do not need to change the encoder's train/eval mode.
        def target_labels(this,source_feature,source_probability,source_index,target_feature):
            with torch.no_grad():
                this.source_f_bank[source_index]=F.normalize(source_feature,dim=1)
                this.source_score_bank[source_index]=source_probability.detach()
                distance=F.normalize(target_feature.detach(),dim=1) @ this.source_f_bank.T
                neighbours=distance.topk(7,dim=1).indices
                scores=this.source_score_bank[neighbours]
                weights=this.get_weight(scores)
                return F.softmax(torch.einsum('ijk,ij->ik',scores,weights),dim=1)
        self.network.get_target_labels=types.MethodType(target_labels,self.network)

    def initialize_banks(self, x, batch=256):
        self.eval()
        with torch.no_grad():
            for start in range(0,len(x),batch):
                features,_=self.network.encoder(x[start:start+batch])
                self.network.source_f_bank[start:start+batch]=F.normalize(features,dim=1)
                self.network.source_score_bank[start:start+batch]=F.softmax(self.network.cls_classifier(features),1)

    def forward(self,x):
        features,_=self.network.encoder(x)
        return self.network.cls_classifier(features)

    def training_loss(self,x,labels,auxiliary,indices,progress):
        source_logits,source_features,target_logits,target_features,*_,pseudo = self.network(x,auxiliary,labels,indices)
        global_loss=self.domain_loss(source_features+.005*torch.randn_like(source_features),
                                     target_features+.005*torch.randn_like(target_features),source_logits,target_logits)
        boost=2*(2/(1+math.exp(-progress))-1)
        return F.cross_entropy(source_logits,labels)+global_loss+boost*F.cross_entropy(target_logits,pseudo.argmax(1))

    def extra_training_state(self):
        return {'feature_bank':self.network.source_f_bank.detach().cpu(),
                'score_bank':self.network.source_score_bank.detach().cpu(),
                'grl_iteration':self.domain_loss.grl.iter_num}

    def restore_training_state(self,state):
        self.network.source_f_bank.copy_(state['feature_bank'])
        self.network.source_score_bank.copy_(state['score_bank'])
        self.domain_loss.grl.iter_num=state['grl_iteration']


def make_model(name, channels, classes, sampling_rate, references, source_count, domains,
               prior, seed, device, legacy_config):
    torch.manual_seed(seed)
    if name=='EEGNet': model=EEGNet(len(channels),sampling_rate,classes,sampling_rate)
    elif name=='TSception':model=TSception(references,channels,sampling_rate,classes,sampling_rate)
    elif name=='RGNN':model=RGNN(len(channels),classes,prior)
    elif name=='MS-MDA':model=MSMDA(len(channels),classes,domains)
    elif name=='NSAL-DGAT':model=NSALDGAT(references,len(channels),classes,source_count,device)
    elif name=='DGCNN':
        from models.DGCNN import DGCNN
        class LockedDGCNN(DGCNN):
            def get_param(self):pass
        model=LockedDGCNN(len(channels),5,classes,k=2,relu_is=1,
                          layers=[128 if len(channels) in [30,32] else 64],dropout_rate=.5)
    elif name=='CDCN':
        from scripts.run_architecture_replication import construct_cdcn
        model,_=construct_cdcn(len(channels),classes,seed,False,legacy_config)
    elif name=='MGGL':
        raise RuntimeError('MGGL requires adequate author-reference information; no substitute is registered')
    else:raise ValueError(name)
    return model.to(device)


def optimizer_for(model,name):
    if name=='NSAL-DGAT':return torch.optim.RMSprop(model.parameters(),lr=.001,weight_decay=.001)
    if name=='DGCNN':return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001,eps=.0001)
    if name=='CDCN':return torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.005,eps=.0001)
    if name=='RGNN':return torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001,eps=.0001)
    return torch.optim.Adam(model.parameters(),lr=.001)


def training_loss(model,name,x,labels,auxiliary,indices,progress,domain_index=None):
    if name=='RGNN':return model.training_loss(x,labels,auxiliary,progress)
    if name=='MS-MDA':return model.training_loss(x,labels,auxiliary,domain_index,progress)
    if name=='NSAL-DGAT':return model.training_loss(x,labels,auxiliary,indices,progress)
    loss=F.cross_entropy(model(x),labels)
    if name=='DGCNN':loss=loss+.01*sum(p.norm() for p in model.parameters())
    return loss
