import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
from cbam import CBAM

class ResNet18_3D(nn.Module):
    def __init__(self, num_classes=2, use_pretrained=True, use_cbam=True):
        
        super(ResNet18_3D, self).__init__()
        
        # Load pretrained ResNet-18
        resnet18_2d = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None)
        
        # Convert 2D convolutions to 3D
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(7,7,7), stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Initialize with 2D weights (repeating along depth)
        if use_pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(resnet18_2d.conv1.weight.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3.0)
                self.bn1.load_state_dict(resnet18_2d.bn1.state_dict())
        
        # Create 3D versions of ResNet blocks
        self.layer1 = self._make_layer(resnet18_2d.layer1, 64, 2, use_cbam)
        self.layer2 = self._make_layer(resnet18_2d.layer2, 128, 2, use_cbam, stride=2)
        self.layer3 = self._make_layer(resnet18_2d.layer3, 256, 2, use_cbam, stride=2)
        self.layer4 = self._make_layer(resnet18_2d.layer4, 512, 2, use_cbam, stride=2)
        
        # Attention modules
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam1 = CBAM(64)
            self.cbam2 = CBAM(128)
            self.cbam3 = CBAM(256)
            self.cbam4 = CBAM(512)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        if not use_pretrained:
            self._initialize_weights()
    
    def _make_layer(self, layer_2d, planes, blocks, use_cbam, stride=1):
        #Create a 3D layer from a 2D ResNet layer.
        layers = []
        
        # First block with possible downsampling
        downsample = None
        if stride != 1 or (isinstance(layer_2d[0].conv1, nn.Conv2d) and layer_2d[0].conv1.in_channels != planes):
            downsample = nn.Sequential(
                nn.Conv3d(
                    layer_2d[0].conv1.in_channels if hasattr(layer_2d[0], 'conv1') else planes // 2,
                    planes * (layer_2d[0].expansion if hasattr(layer_2d[0], 'expansion') else 1),
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * (layer_2d[0].expansion if hasattr(layer_2d[0], 'expansion') else 1)),
            )
        
        # First block
        layers.append(BasicBlock3D(
            layer_2d[0],
            planes,
            stride,
            downsample,
            use_cbam
        ))
        
        # Remaining blocks
        for i in range(1, blocks):
            layers.append(BasicBlock3D(
                layer_2d[i],
                planes,
                use_cbam=use_cbam
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        #Initialize weights for Conv3d and BatchNorm3d layers.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        in_ch = x.shape[1]
        if in_ch != self.conv1.in_channels:
            # Create a new conv with matching in_channels and same other params
            new_conv = nn.Conv3d(
                in_ch,
                self.conv1.out_channels,
                kernel_size=self.conv1.kernel_size,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                bias=(self.conv1.bias is not None),
            )
            
            try:
                with torch.no_grad():
                    old_w = self.conv1.weight.data
                    # old_w shape: [out, in_old, kD, kH, kW]
                    in_old = old_w.size(1)
                    if in_old == in_ch:
                        new_conv.weight.data.copy_(old_w)
                    else:
                        # Compute mean over input channels and tile to new in_ch
                        mean_w = old_w.mean(dim=1, keepdim=True)  # [out,1,kD,kH,kW]
                        adapted = mean_w.repeat(1, in_ch, 1, 1, 1)
                        new_conv.weight.data.copy_(adapted)
                    if self.conv1.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data.copy_(self.conv1.bias.data)
            except Exception:
                pass

            new_conv = new_conv.to(next(self.parameters()).device)
            self.conv1 = new_conv
            try:
                print(f"[model_classifier] Adapted conv1 from {in_old if 'in_old' in locals() else 'unknown'} -> {in_ch} input channels")
            except Exception:
                pass

        # Initial conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers with optional CBAM
        if self.use_cbam:
            x = self.layer1(x)
            x = self.cbam1(x)
            x = self.layer2(x)
            x = self.cbam2(x)
            x = self.layer3(x)
            x = self.cbam3(x)
            x = self.layer4(x)
            x = self.cbam4(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class BasicBlock3D(nn.Module):
    #3D version of the BasicBlock used in ResNet-18.
    expansion = 1
    
    def __init__(self, block_2d, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock3D, self).__init__()
        
        # Convert 2D conv to 3D
        self.conv1 = self._conv2d_to_3d(block_2d.conv1, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = self._conv2d_to_3d(block_2d.conv2)
        self.bn2 = nn.BatchNorm3d(planes)
        
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        
        # Copy weights from 2D to 3D
        with torch.no_grad():
            if hasattr(block_2d, 'conv1'):
                self._copy_conv_weights(block_2d.conv1, self.conv1)
                self._copy_conv_weights(block_2d.conv2, self.conv2)
                
                if hasattr(block_2d, 'bn1'):
                    self.bn1.load_state_dict(block_2d.bn1.state_dict())
                    self.bn2.load_state_dict(block_2d.bn2.state_dict())
        
        # CBAM
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(planes * self.expansion)
    
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
        
        if self.use_cbam:
            out = self.cbam(out)
        
        return out
    
    def _conv2d_to_3d(self, conv2d, stride=1):
        #Convert a 2D convolution to 3D.
        if conv2d is None:
            return None
            
        kernel_size = conv2d.kernel_size[0] if isinstance(conv2d.kernel_size, (tuple, list)) else conv2d.kernel_size
        padding = conv2d.padding[0] if isinstance(conv2d.padding, (tuple, list)) else conv2d.padding
        
        return nn.Conv3d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(stride, stride, stride),
            padding=(padding, padding, padding),
            bias=conv2d.bias is not None
        )
    
    def _copy_conv_weights(self, conv2d, conv3d):
        #Copy weights from 2D to 3D convolution.
        if conv2d is None or conv3d is None:
            return
            
        with torch.no_grad():
            # Repeat weights along depth dimension
            weight_2d = conv2d.weight.data
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, weight_2d.size(2), 1, 1) / weight_2d.size(2)
            conv3d.weight.data.copy_(weight_3d)
            
            if conv2d.bias is not None and conv3d.bias is not None:
                conv3d.bias.data.copy_(conv2d.bias.data)

class NoduleClassifier(nn.Module):
    #Wrapper for the nodule classification model with Grad-CAM support.
    def __init__(self, num_classes=2, use_pretrained=True, use_cbam=True):
        super(NoduleClassifier, self).__init__()
        self.model = ResNet18_3D(num_classes, use_pretrained, use_cbam)
        self.features = None
        self.gradients = None
        
        # Register hooks for Grad-CAM
        target_layer = self.model.layer4[-1].conv2
        target_layer.register_forward_hook(self.save_features)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def forward(self, x):
        return self.model(x)
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.features

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with CBAM
    model_cbam = NoduleClassifier(use_cbam=True).to(device)
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    out = model_cbam(x)
    print("Model with CBAM output shape:", out.shape)
    
    # Test without CBAM
    model_no_cbam = NoduleClassifier(use_cbam=False).to(device)
    out = model_no_cbam(x)
    print("Model without CBAM output shape:", out.shape)
