import mlconfig
import math
from torch import nn
import torch.nn.functional as F
import torch
        
class ConvBNReLU(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNReLU, self).__init__(*layers)

class DepthwiseConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(DepthwiseConv, self).__init__(*layers)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class eSEModule(nn.Module):
    def __init__(self, channel):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = h_sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x

class _OSA_module(nn.Module):
    def __init__(self, in_ch, stage_ch, concat_ch, layer_per_block, SE=False, identity=False, depthwise=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.layers = nn.ModuleList()
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = ConvBNReLU(in_channel, stage_ch, kernel_size=1, stride=1, padding=0)
            in_channel = stage_ch
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(DepthwiseConv(in_channel, stage_ch, kernel_size=3, stride=1, padding=1))
            else:
                self.layers.append(ConvBNReLU(in_channel, stage_ch, kernel_size=3, stride=1, padding=1))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = ConvBNReLU(in_channel, concat_ch, kernel_size=1, stride=1, padding=0)

        self.ese = eSEModule(concat_ch)

    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt
    
class _OSA_stage(nn.Sequential):
    
    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage, layer_per_block, stage_num, SE=False, depthwise=False):
        layers = []
        if not stage_num == 2:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        if block_per_stage != 1:
            SE = False
        layers.append(_OSA_module(in_ch, stage_ch, concat_ch, layer_per_block, SE, depthwise=depthwise))
        for i in range(block_per_stage - 1):
            if i != block_per_stage - 2:  # last block
                SE = False
            layers.append(_OSA_module(concat_ch, stage_ch, concat_ch, layer_per_block, SE, identity=True, depthwise=depthwise))
              
        super(_OSA_stage, self).__init__(*layers)
        
class VoVNetV2(nn.Module):
    def __init__(self):
        super(VoVNetV2, self).__init__()
        """VoVNet39_dw_eSE"""
        stem_ch = [64, 64, 128]
        config_stage_ch = [128, 160, 192, 224]
        config_concat_ch = [256, 512, 768, 1024]
        block_per_stage = [1, 1, 2, 2]
        layer_per_block = 5
        SE = True
        depthwise = True
        """VoVNet19_dw_eSE"""
#         stem_ch = [64, 64, 64]
#         config_stage_ch = [128, 160, 192, 224]
#         config_concat_ch = [256, 512, 768, 1024]
#         block_per_stage = [1, 1, 1, 1]
#         layer_per_block = 3
#         SE = True
#         depthwise = True
        in_ch_list = stem_ch[-1:] + config_concat_ch[:-1]
        
        self.stem1 = ConvBNReLU(3, stem_ch[0], kernel_size=3, stride=2, padding=2)
        if depthwise:
            self.stem2 = DepthwiseConv(stem_ch[0], stem_ch[1], kernel_size=3, stride=1, padding=1)
            self.stem3 = DepthwiseConv(stem_ch[1], stem_ch[2], kernel_size=3, stride=2, padding=1)
        else:
            self.stem2 = ConvBNReLU(stem_ch[0], stem_ch[1], kernel_size=3, stride=1, padding=1)
            self.stem3 = ConvBNReLU(stem_ch[1], stem_ch[2], kernel_size=3, stride=2, padding=1)
        
        # OSA stages
        self.osa_layers = nn.ModuleList()
        self.block_layers = nn.ModuleList()
        
        for i in range(4):  # num_stages
            self.osa_layers.append(_OSA_stage(in_ch_list[i], config_stage_ch[i], config_concat_ch[i], block_per_stage[i], layer_per_block, i + 2, SE, depthwise))
            if i != 0:
                self.block_layers.append(ConvBNReLU(config_concat_ch[i], 256, kernel_size=1, stride=1, padding=0))
        
        
    def forward(self, x):
        outputs = []
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem3(x)
        for i, layer in enumerate(self.osa_layers):
            x = layer(x)
            if i != 0:
                outs = self.block_layers[i-1](x)
                outputs.append(outs)
        return outputs
        
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
    
class FCOSHead(nn.Module):
    def __init__(self, in_channel, num_classes, n_conv):
        super(FCOSHead, self).__init__()

        cls_tower = []
        vertex_tower = []

        for i in range(n_conv):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())

            vertex_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            vertex_tower.append(nn.GroupNorm(32, in_channel))
            vertex_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.vertex_tower = nn.Sequential(*vertex_tower)
        self.cls_pred = nn.Conv2d(in_channel, num_classes, 3, padding=1)
        self.vertex_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)
        prior_bias = -math.log((1-0.01)/0.01)
        nn.init.constant_(self.cls_pred.bias, prior_bias)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, inputs):
        logits = []
        vertexes = []
        centers = []

        for feat, scale in zip(inputs, self.scales):
            cls_out = self.cls_tower(feat)
            
            clss_out = self.cls_pred(cls_out)
            center_out = self.center_pred(cls_out)

            logits.append(clss_out)
            centers.append(center_out)

            vertex_out = self.vertex_tower(feat)
            vertex_out = torch.exp(scale(self.vertex_pred(vertex_out)))

            vertexes.append(vertex_out)
        return logits, vertexes, centers
        
@mlconfig.register
class VoVNetV2_FCOS(nn.Module):

    def __init__(self, num_classes=2):
        super(VoVNetV2_FCOS, self).__init__()
#         self.features = nn.ModuleList(MobileNet.get_layers())
        self.features = VoVNetV2()
        
#         self.inner_conv = nn.ModuleList([nn.Conv2d(256, 256, 1), nn.Conv2d(512, 256, 1), nn.Conv2d(1024, 256, 1)])
        
        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv_out6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
        
#         self.out_conv = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1), nn.Conv2d(256, 256, 3, padding=1), nn.Conv2d(256, 256, 3, padding=1)])
        
        self.head = FCOSHead(256, num_classes, 4)
        
        self._initialize_weights()

    def forward(self, x):
#         sources = []
#         for i, v in enumerate(self.features):
#             x = v(x)
#             if i in [5, 11, 13]:
#                 sources.append(x)
    
#         for i, v in enumerate(self.inner_conv):
#             sources[i] = v(sources[i])
        
        sources = self.features(x)
        C3_block, C4_block, P5 = sources
        
        P4 = C4_block + F.interpolate(P5, size=(int(C4_block.shape[-2]), int(C4_block.shape[-1])), mode='nearest')
        P3 = C3_block + F.interpolate(P4, size=(int(C3_block.shape[-2]), int(C3_block.shape[-1])), mode='nearest')
        
        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        
        
        cls_preds, vertex_preds, center_preds = self.head([P3, P4, P5, P6, P7])
        
        return cls_preds, vertex_preds, center_preds
        
        
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                