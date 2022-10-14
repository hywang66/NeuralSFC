import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import utils.cfg
from torch.nn import ReLU
from torch_geometric.data import Batch, Data
from torch_geometric.nn import DeepGCNLayer
from utils.functions import construst_dual_edge_index


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DualGraphEncoder(nn.Module):

    def __init__(self, n_layers=[2, 2, 2, 2], n_channels=3, n_base_embeddings=64, norm_layer=None): 
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        

        self.conv_dual = nn.Conv2d(n_channels, n_base_embeddings, 2, stride=2)
        self.norm_dual = self._norm_layer(n_base_embeddings)
        self.relu = nn.ReLU(inplace=True)

        self.layers = []
        self.inplanes = n_base_embeddings
        for i, nl in enumerate(n_layers):
            self.layers.append(self._make_layer(ResBlock, n_base_embeddings * (2 ** i), nl))
        
        self.layers = nn.Sequential(*self.layers)

    def _make_layer(self, block, planes, blocks):
        downsample = None
        norm_layer = self._norm_layer

        if self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, downsample=downsample, norm_layer=norm_layer))
        
        for _ in range(1, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))
            
        self.inplanes = planes
        
        return nn.Sequential(*layers)

    def forward(self, img):
        x = self.conv_dual(img)
        x = self.norm_dual(x)
        x = self.relu(x)
        x = self.layers(x)
        
        return x # [bs, n_base_embeddings*8, 16, 16]

        
class LineGraphConverter(nn.Module):
    def __init__(self, grid_size=16, n_embeddings=512):
        super().__init__()
        self.edge_h_pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.edge_v_pool = nn.AvgPool2d(kernel_size=(2, 1), stride=(1, 1))
        # self.edge_index = construst_dual_edge_index(grid_size, grid_size)
        self.edge_index = None

    def forward(self, x):
        bs, n_embeddings = x.shape[:2]
        grid_size = x.shape[-1]

        if self.edge_index is None:
            self.edge_index = construst_dual_edge_index(grid_size, grid_size)
            self.edge_index = self.edge_index.to(x.device)

        edges_h = self.edge_h_pool(x).view(bs, n_embeddings, -1) # [bs, n_base_embeddings*8, 16, 15]
        edges_v = self.edge_v_pool(x).view(bs, n_embeddings, -1) # [bs, n_base_embeddings*8, 15, 16]
        edges_h_v = torch.cat([edges_h, edges_v], dim=2).permute(0, 2, 1).contiguous() # [bs, 480, n_embeddings]

        batch_list = [Data(x=x, edge_index=self.edge_index) for x in edges_h_v]
        batch = Batch.from_data_list(batch_list).to(x.device)

        return batch


class GrpahScalarRegressor(nn.Module):

    def __init__(self, n_embeddings=512, n_layers: int = 6) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.embed_weights = nn.Linear(1, n_embeddings)
        self.convs = nn.ModuleList()
        self.bn0 = gnn.BatchNorm(n_embeddings)
        self.bnw = gnn.BatchNorm(n_embeddings)
        self.bns = nn.ModuleList()

        for _ in range(n_layers - 1):
            self.convs.append(gnn.GCNConv(n_embeddings, n_embeddings))
            self.bns.append(gnn.BatchNorm(n_embeddings))

        # add the final gcnconv
        self.convs.append(gnn.GCNConv(n_embeddings, n_embeddings))
        self.linear = nn.Linear(n_embeddings, 1)

    def forward(self, graph_batch: Batch, weights: torch.Tensor) -> torch.Tensor:
        # assuming that graph_batch and weights are from a same graph batch,
        # with node features n_embeddings and 1, respectively
        if weights.size(-1) != 1:
            weights = weights.view(-1, 1)

        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        embed_weights = self.embed_weights(weights)
        x = self.bn0(x)
        embed_weights = self.bnw(embed_weights)
        x = x + embed_weights
        x = x.relu()

        for i in range(self.n_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = x.relu()

        # apply the last conv
        x = self.convs[-1](x, edge_index)

        # readout layer
        x = gnn.global_mean_pool(x, batch) # [bs, n_embeddings]
        
        scalar = self.linear(x).squeeze()
        
        return scalar


class ScalarEvaluator(nn.Module):
    def __init__(
        self, 
        n_encoder_layers=[2, 2, 2, 2], 
        n_img_channels=1, 
        n_embeddings=512, 
        encoder_norm_layer=None,
        n_regressor_layers=6,
    ) -> None:

        super().__init__()

        self.image_encoder = DualGraphEncoder(
            n_layers=n_encoder_layers, 
            n_channels=n_img_channels,
            n_base_embeddings=n_embeddings // 8,
            norm_layer=encoder_norm_layer
        )

        self.line_graph_converter = LineGraphConverter(n_embeddings=n_embeddings)

        if utils.cfg.cfg_global is None or utils.cfg.cfg_global.graph_merge == 'concat':
            self.scalar_regressor = GrpahScalarRegressor(n_embeddings=n_embeddings, n_layers=n_regressor_layers)
        else:
            raise NotImplementedError
        print(f'Using {self.scalar_regressor.__class__.__name__}')

    def forward(self, img: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        dual_graph_emb = self.image_encoder(img)
        line_graph_emb = self.line_graph_converter(dual_graph_emb)
        scalar = self.scalar_regressor(line_graph_emb, weights)
        return scalar


class NACEvaluator(ScalarEvaluator):
    # Negative Autocorrelation Evaluator
    def __init__(
        self, 
        n_encoder_layers=[2, 2, 2, 2], 
        n_img_channels=1, 
        n_embeddings=512, 
        encoder_norm_layer=None
    ):
        super().__init__(n_encoder_layers, n_img_channels, n_embeddings, encoder_norm_layer)
        self.scalar_regressor.linear.bias.data = torch.tensor([0.0])

    def forward(self, img: torch.Tensor, weights: torch.Tensor):
        return -super().forward(img, weights).sigmoid()


class GrpahWeightRegressor(nn.Module):

    def __init__(self, n_embeddings=512) -> None:
        super().__init__()
        self.conv1 = gnn.GCNConv(n_embeddings, n_embeddings)
        self.conv2 = gnn.GCNConv(n_embeddings, n_embeddings)
        self.conv_out = gnn.GCNConv(n_embeddings, 1)
        self.bnw = gnn.BatchNorm(n_embeddings)
        self.bn1 = gnn.BatchNorm(n_embeddings)
        self.bn2 = gnn.BatchNorm(n_embeddings)
        self.linear = nn.Linear(n_embeddings, 1)

        

    def forward(self, graph_batch: Batch) -> torch.FloatTensor:
        # assuming that graph_batch and weights are from a same graph batch,
        # with node features n_embeddings and 1, respectively
        bs = graph_batch.num_graphs
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        weights = self.conv_out(x, edge_index).view(bs, -1)
        return weights # [bs, H x W]


class GrpahWeightRegressor_large(nn.Module):

    def __init__(self, n_embeddings=512) -> None:
        super().__init__()
        self.bn1 = gnn.BatchNorm(n_embeddings)
        self.bn2 = gnn.BatchNorm(n_embeddings)
        self.bn3 = gnn.BatchNorm(n_embeddings)
        self.bn4 = gnn.BatchNorm(n_embeddings)
        self.bn5 = gnn.BatchNorm(n_embeddings)
        self.conv1 = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)
        self.conv2 = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)
        self.conv3 = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)
        self.conv4 = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)
        self.conv5 = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)
        self.conv6 = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)
        self.linear = nn.Linear(n_embeddings, 1)


    def forward(self, graph_batch: Batch) -> torch.FloatTensor:
        # assuming that graph_batch and weights are from a same graph batch,
        # with node features n_embeddings and 1, respectively
        bs = graph_batch.num_graphs
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = x.relu()
        weights = self.linear(x).view(bs, -1)
        return weights # [bs, H x W]


class GrpahWeightRegressor_res_gat(nn.Module):

    def __init__(self, n_embeddings=512) -> None:
        super().__init__()
        self.linear = nn.Linear(n_embeddings, 1)
        num_layers = 7
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = gnn.GATConv(n_embeddings, n_embeddings // 8, 8)

            norm = gnn.BatchNorm(n_embeddings)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)
        

    def forward(self, graph_batch: Batch) -> torch.FloatTensor:
        # assuming that graph_batch and weights are from a same graph batch,
        # with node features n_embeddings and 1, respectively
        bs = graph_batch.num_graphs
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        weights = self.linear(x).view(bs, -1)
        return weights # [bs, H x W]


class WeightGenerator(nn.Module):
    def __init__(
        self, 
        n_encoder_layers=[2, 2, 2, 2], 
        n_img_channels=1, 
        n_embeddings=512, 
        encoder_norm_layer=None,
        large_wg=False,
    ) -> None:

        super().__init__()

        self.image_encoder = DualGraphEncoder(
            n_layers=n_encoder_layers, 
            n_channels=n_img_channels,
            n_base_embeddings=n_embeddings // 8,
            norm_layer=encoder_norm_layer
        )

        self.line_graph_converter = LineGraphConverter(n_embeddings=n_embeddings)

        if large_wg == True:
            self.weight_regressor = GrpahWeightRegressor_large(n_embeddings=n_embeddings)
        elif large_wg == False:
            self.weight_regressor = GrpahWeightRegressor(n_embeddings=n_embeddings)
        elif large_wg == 'res_gat':
            self.weight_regressor = GrpahWeightRegressor_res_gat(n_embeddings=n_embeddings)
        else:
            raise NotImplementedError
        
        print(large_wg)
        print(f'Using {type(self.weight_regressor)}.')

    def forward(self, img: torch.Tensor):
        dual_graph_emb = self.image_encoder(img)
        line_graph_emb = self.line_graph_converter(dual_graph_emb)
        weights = self.weight_regressor(line_graph_emb)
        return weights # [bs, n]


