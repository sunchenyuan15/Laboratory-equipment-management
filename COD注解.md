# 创建dataloader
```python
val_cfg = [Dataset.Config(datapath=f'{root}/test' , mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K']]#datapath=f'{root}'/test/{i}   mode='test'
val_data = [Dataset.Data(v) for v in val_cfg]
val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=4) for v in val_data] 

class Config(object):
    def __init__(self, **kwargs):
        if kwargs.get('label_dir') is None:
            kwargs['label_dir'] = 'Scribble'
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])
            # self.std, self.mean = np.array([0.1861761914527739, 0.19748777412623036, 0.2032849354904543])[None,None]*255, np.array([0.3320486163733052, 0.432231354815684, 0.449829585669272])[None,None]*255

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_name = cfg.datapath.split('/')[-1]
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            if cfg.mode == 'train':
              for line in lines:
                  imagepath = cfg.datapath +'/'+cfg.mode+ '/Image/' + line.strip() + '.png'
                  maskpath  = cfg.datapath + '/'+cfg.mode+f'/{cfg.label_dir}/'  + line.strip() + '.png'
                  self.samples.append([imagepath, maskpath])
            else:
              for line in lines:
                  imagepath = cfg.datapath + '/Image/' + line.strip() + '.png'
                  maskpath  = cfg.datapath +'/GT/'  + line.strip() + '.png'
                  self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.RandomHorizontalFlip(),
                                                    transform.RandomCrop(320, 320),
                                                    transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose(transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform.Resize(320, 320),
                                                    transform.ToTensor()
                                                )
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        if self.cfg.mode == 'train':
            image, mask         = self.transform(image, mask)
            mask[mask == 0.] = 255.
            mask[mask == 2.] = 0.
        else:
            image, _         = self.transform(image, mask)
            mask = torch.from_numpy(mask.copy()).permute(2,0,1)
            mask = mask.mean(dim=0, keepdim=True)
            mask /= 255
        # print(image.max(), image.min())
        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)
```

# **使用异步预加载数据：**
```python
prefetcher = DataPrefetcher(loader)
image, mask = prefetcher.next()

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)#使用iter()函数将可迭代对象转换为迭代器。迭代器是一种对象，可以使用next()函数逐个访问可迭代对象的元素。
        self.stream = torch.cuda.Stream()
        self.preload()
    #类的目的是提高数据加载和GPU计算之间的并行性，以加速训练过程。
    #通过异步预加载数据，可以减少数据加载和GPU计算之间的等待时间，从而更有效地利用GPU资源。
    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
```
# 动态调整学习率
```python
niter = epoch * db_size + batch_idx
lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)#动态调整学习率
#动量是深度学习中一种优化算法，通过考虑之前的梯度更新信息，使参数更新具有一定的惯性，加速梯度下降的收敛速度，并帮助模型更快地达到最优解。 

def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum
```
# **随机数据变换：**
```python
pre_transform = get_transform(ops)
image_tr = pre_transform(image)

def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

class Flip:
    def __init__(self, flip):
        self.flip = flip
    def __call__(self, img):
        if self.flip==0:
            return img.flip(-1)
        else:
            return img.flip(-2)
    
class Translate:
    def __init__(self, fct):
        '''Translate offset factor'''
        drct = np.random.randint(0, 4)
        self.signed_x = drct>=2 or -1#True、False 或 -1 中的一个 drct为负数则赋值为-1
        self.signed_y = drct%2 or -1#偶数-1 ；奇数 True
        self.fct = fct
    def __call__(self, img):
        angle = 0
        scale = 1
        h, w = img.shape[-2:]
        h, w = int(h*self.fct), int(w*self.fct)
        return affine(img, angle, (h*self.signed_y,w*self.signed_x), scale, shear=0, interpolation=InterpolationMode.BILINEAR)

class Crop:
    def __init__(self, H, W):
        '''keep the relative ratio for offset'''
        self.h = H
        self.w = W
        self.xm  = np.random.uniform()
        self.ym  = np.random.uniform()
        # print(self.xm, self.ym)
    def __call__(self, img):
        H,W = img.shape[-2:]
        sh = int(self.h*H)
        sw = int(self.w*W)
        ymin = int((H-sh+1)*self.ym)
        xmin = int((W-sw+1)*self.xm)
        img = img[..., ymin:ymin+ sh, xmin:xmin+ sw]
        img = F.interpolate(img, size=(H,W), mode='bilinear', align_corners=False)
        return img
```
# **ResNet-50：**
```python
bk_stage1, bk_stage2, bk_stage3, bk_stage4, bk_stage5 = self.bkbone(x)


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)
    
    def initialize(self):
        weight_init(self)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/public/home/shenjl/Pythoncode/COD/COD_main/assets/resnet50-19c8e357.pth'), strict=False)

```
# **金字塔池化（可获取不同尺度的上下文信息）：**
```python
self.pyramid_pooling = PyramidPooling(2048, 64)

f_c3 = self.pyramid_pooling(bk_stage5)
#这里bk_staged5.size() = torch.Size([16, 2048, 10, 10])


class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            #conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
    
    def initialize(self):
        weight_init(self)

def weight_init(module):#根据不同的参数子模块，采用不同的权重初始化方法  有助于模型的收敛和性能的提升。
    for n, m in module.named_children():#使用递归的方式遍历 module 中的所有子模块，并根据子模块的类型进行不同的权重初始化操作。
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            m.initialize()

class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel*2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]#自适应平均池化将输入特征图分别池化为不同尺度的特征图   从而捕捉不同尺度下的上下文信息
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)#F.adaptive_avg_pool2d(x, 1):返回的最后两个维度大小是[1, 1]
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)#[2, 2]
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
        
    def initialize(self):
        weight_init(self)
```
# **逻辑语义关系(LSR)模块：**
```python
self.rfb = nn.ModuleList([
     RFB_modified(1024, 64),
     RFB_modified(2048, 64)
])

f_c2 = self.rfb[1](bk_stage5)#逻辑语义关系模块
#这里bk_staged5.size() = torch.Size([16, 2048, 10, 10])

# Revised from: PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI20
# https://github.com/DengPingFan/PraNet

# LSR模块从4个分支中提取语义特征。每个分支包含一系列具有不同核大小和扩展率的卷积层，代表不同的接受域。
# 然后，我们将所有分支的信息整合起来，利用更广泛的接受野来获取全面的语义信息，以确定真实的前景和背景。

class RFB_modified(nn.Module):
    """ logical semantic relation (LSR) """
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            basicConv(in_channel, out_channel, 1, relu=False),
        )
        self.branch1 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch2 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.branch3 = nn.Sequential(
            basicConv(in_channel, out_channel, 1),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, k=7, p=3),
            basicConv(out_channel, out_channel, 3, p=7, d=7, relu=False)
        )
        self.conv_cat = basicConv(4*out_channel, out_channel, 3, p=1, relu=False)
        self.conv_res = basicConv(in_channel, out_channel, 1, relu=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
        
    def initialize(self):
        weight_init(self)

```
# **特征融合模块(FFM)：**
```python
self.fusion = nn.ModuleList([
    FFM(64),
    FFM(64),
    FFM(64),
    FFM(64)
])

fused3 = self.fusion[2](f_c2, fused3)
#f_c2和fused3的size都是torch.Size([16, 64, 12, 12])

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        #out = torch.cat((x_1, x_2), dim=1)
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)

```
# **局部上下文对比模块(LCC)：**
```python
self.contrast = nn.ModuleList([
    Contrast_Block_Deep(64),
    Contrast_Block_Deep(64)
])
f_t2 = self.contrast[1](f_t2)#局部上下文对比模块  维度和大小都没变
#f_t2.size() = torch.Size([16, 64, 40, 40])

########################################### CoordAttention #########################################
# Revised from: Coordinate Attention for Efficient Mobile Network Design, CVPR21
# https://github.com/houqb/CoordAttention
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

    def initialize(self):
        weight_init(self)

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

    def initialize(self):
        weight_init(self)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
    #坐标注意力机制旨在增强模型对输入特征图中不同位置的重要性。它通过考虑特征图的坐标信息，使模型能够更加关注不同位置的特征，从而提升模型的感知能力和表达能力。
    def forward(self, x):#坐标注意力机制能够对输入特征图的通道、高度和宽度进行注意力调节，以提取更加有用和相关的特征。
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
    
    def initialize(self):
        weight_init(self)

####################################### Contrast Texture ###########################################
class Contrast_Block_Deep(nn.Module):
    """ local-context contrasted (LCC) """
    def __init__(self, planes, d1=4, d2=8):
        super(Contrast_Block_Deep, self).__init__()
        self.inplanes = int(planes)
        self.outplanes = int(planes / 2)

        self.local_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_1 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d1, dilation=d1)

        self.local_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=1, dilation=1)
        self.context_2 = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=3, stride=1, padding=d2, dilation=d2)

        self.bn1 = nn.BatchNorm2d(self.outplanes)
        self.bn2 = nn.BatchNorm2d(self.outplanes)


        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.ca = nn.ModuleList([
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes),
            CoordAtt(self.outplanes, self.outplanes)
        ])


    def forward(self, x):
        local_1 = self.local_1(x)
        local_1 = self.ca[0](local_1)
        context_1 = self.context_1(x)
        context_1 = self.ca[1](context_1)
        ccl_1 = local_1 - context_1
        ccl_1 = self.bn1(ccl_1)
        ccl_1 = self.relu1(ccl_1)

        local_2 = self.local_2(x)
        local_2 = self.ca[2](local_2)
        context_2 = self.context_2(x)#1和2之间的填充和膨胀的参数不一样
        context_2 = self.ca[3](context_2)
        ccl_2 = local_2 - context_2
        ccl_2 = self.bn2(ccl_2)
        ccl_2 = self.relu2(ccl_2)

        out = torch.cat((ccl_1, ccl_2), 1)
        #输入和输出的维度是一样的
        return out

    def initialize(self):
        weight_init(self)

```
# **跨聚合模块(CAM)：**
```python
self.aggregation = nn.ModuleList([
    CAM(64),
    CAM(64)
])

a2 = self.aggregation[1](a2, f_t2)
#a2的size：torch.Size([16, 64, 20, 20])； f_t2的size：torch.Size([16, 64, 40, 40])

# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)
    #通过跨通道和跨空间的聚合操作，跨聚合模块能够促进不同层级和尺度的特征之间的信息交互和融合，从而提升模型的表达能力和对复杂特征的感知能力
    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)

```
# **显著性结构一致性损失(SaliencyStructureConsistency)：**
```python
loss_ssc = (SaliencyStructureConsistency(out2_s, out2_scale.detach(), 0.85) * (w_l2g + 1) + SaliencyStructureConsistency(out2_s.detach(), out2_scale, 0.85) * (1 - w_l2g)) if sl else 0
#out_s:[16,1,96,96]; out2_scale:[16,1,96,96]

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)#这里输入输出大小不变
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)#平方操作
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq#前面一项是先平方再池化，后面是先池化再激活
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    #函数返回的结果是 (1 - SSIM) / 2 的张量，表示两个图像之间的结构相似性损失（取值范围为0到1）。
    return torch.clamp((1 - SSIM) / 2, 0, 1)


def SaliencyStructureConsistency(x, y, alpha):
    ssim = torch.mean(SSIM(x,y))#结构相似性，用来评价两个图像之间的相似性，越接近一越相似
    l1_loss = torch.mean(torch.abs(x-y))
    loss_ssc = alpha*ssim + (1-alpha)*l1_loss
    return loss_ssc

```
# **局部显著性相关损失(LSC)：**
```python
######   local saliency coherence loss (scale to realize large batchsize)  ######
#主要就是计算原图和预测结果之间的某种局部一致性损失
image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
sample = {'rgb': image_}
out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
#image：[16,3,320,320]; image_: [16,3,80,80]; out2_: [16,1,80,80]

"""
The following code is modified from the file in https://github.com/siyueyu/SCWSSOD/blob/f8650567cbbc8df5bf6edc32a633c47a885574cd/lscloss.py.
Credit for them.
"""
class FeatureLoss(torch.nn.Module):
    """
    This loss function based on the following paper.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    """
    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param mask_dst: (optional) Destination mask.
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device
        #下面这个assert保证了输入图片和预测结果之间的大小是一致的
        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)#刚才的核是对原图片进行展开操作，现在对预测结果进行展开操作
        y_hat_unfolded = torch.abs(y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred, width_pred) - y_hat_unfolded)

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True))
        #先将核和预测结果的展开式相乘，然后塑形为[16,1,121,80,80]的大小，然后在第三(2)个维度上进行求和，得到[16,1,1,80,80]

        out = {
            'loss': loss.mean(),
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = FeatureLoss._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                    # feature = LocalSaliencyCoherence._downsample(
                    #     feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    # )
                feature /= sigma
                features.append(feature)#原特征和利用高宽生成的网格特征叠加在一起
            features = torch.cat(features, dim=1)
            kernel = weight * FeatureLoss._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = FeatureLoss._unfold(features, radius)#对给定的特征进行展开操作
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()#对上述张量在维度 1（通道维度）上进行求和操作，得到一个形状为 (N, 1, H, W) 的张量。keepdim=True 参数表示保持求和结果的维度为 1。
        # kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis

```
# **日志：**
```python
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")#定义日志消息的格式
    logger = logging.getLogger(name)#创建一个日志记录器
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")#文件处理器，用来将日志消息写到指定的文件中
    fh.setFormatter(formatter)#将之前设置的消息格式应用到文件处理器中
    logger.addHandler(fh)#将文件处理器添加到日志记录器中，方便将日志消息写入到文件中

    sh = logging.StreamHandler()#流处理器，将消息输入到终端
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger

logger = Get_logger(args.Dir_Log + log_name)
```
# **日志：**
```python
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")#定义日志消息的格式
    logger = logging.getLogger(name)#创建一个日志记录器
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")#文件处理器，用来将日志消息写到指定的文件中
    fh.setFormatter(formatter)#将之前设置的消息格式应用到文件处理器中
    logger.addHandler(fh)#将文件处理器添加到日志记录器中，方便将日志消息写入到文件中

    sh = logging.StreamHandler()#流处理器，将消息输入到终端
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger

logger = Get_logger(args.Dir_Log + log_name)
```
# **日志：**
```python
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")#定义日志消息的格式
    logger = logging.getLogger(name)#创建一个日志记录器
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")#文件处理器，用来将日志消息写到指定的文件中
    fh.setFormatter(formatter)#将之前设置的消息格式应用到文件处理器中
    logger.addHandler(fh)#将文件处理器添加到日志记录器中，方便将日志消息写入到文件中

    sh = logging.StreamHandler()#流处理器，将消息输入到终端
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger

logger = Get_logger(args.Dir_Log + log_name)
```
# **日志：**
```python
def Get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s] %(message)s")#定义日志消息的格式
    logger = logging.getLogger(name)#创建一个日志记录器
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")#文件处理器，用来将日志消息写到指定的文件中
    fh.setFormatter(formatter)#将之前设置的消息格式应用到文件处理器中
    logger.addHandler(fh)#将文件处理器添加到日志记录器中，方便将日志消息写入到文件中

    sh = logging.StreamHandler()#流处理器，将消息输入到终端
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger

logger = Get_logger(args.Dir_Log + log_name)
```
# 









