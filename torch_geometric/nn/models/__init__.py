from .mlp import MLP
from .basic_gnn import GCN, GraphSAGE, GIN, GAT, PNA
from .jumping_knowledge import JumpingKnowledge
from .node2vec import Node2Vec
from .deep_graph_infomax import DeepGraphInfomax
from .autoencoder import InnerProductDecoder, GAE, VGAE, ARGA, ARGVA
from .signed_gcn import SignedGCN
from .re_net import RENet
from .graph_unet import GraphUNet
from .schnet import SchNet
from .dimenet import DimeNet
from .explainer import Explainer, to_captum
from .gnn_explainer import GNNExplainer
from .metapath2vec import MetaPath2Vec
from .deepgcn import DeepGCNLayer, MyDeepGCNLayer
from .tgn import TGNMemory
from .label_prop import LabelPropagation
from .correct_and_smooth import CorrectAndSmooth
from .attentive_fp import AttentiveFP
from .rect import RECT_L
from .linkx import LINKX
from .lightgcn import LightGCN
from .mask_label import MaskLabel

__all__ = [
    'MLP',
    'GCN',
    'GraphSAGE',
    'GIN',
    'GAT',
    'PNA',
    'JumpingKnowledge',
    'Node2Vec',
    'DeepGraphInfomax',
    'InnerProductDecoder',
    'GAE',
    'VGAE',
    'ARGA',
    'ARGVA',
    'SignedGCN',
    'RENet',
    'GraphUNet',
    'SchNet',
    'DimeNet',
    'Explainer',
    'to_captum',
    'GNNExplainer',
    'MetaPath2Vec',
    'DeepGCNLayer',
    'TGNMemory',
    'LabelPropagation',
    'CorrectAndSmooth',
    'AttentiveFP',
    'RECT_L',
    'LINKX',
    'LightGCN',
    'MaskLabel',
    'MyDeepGCNLayer',
]

classes = __all__
