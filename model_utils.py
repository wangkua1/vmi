import torch
import matplotlib.pylab as plt
import numpy as np
import os
import yaml
from tqdm import tqdm
import itertools
import json
from types import SimpleNamespace

# local imports
import nets
 
from classify_mnist import Net, ResNetCls, ResNetCls1, VGG, get_model


def _load_cls(dataset, cls_path, device):
    with open(os.path.join(os.path.split(cls_path)[0], 'args.json'), 'r') as f:
        model_args = json.load(f)
    # if 'model' in model_args:
    #     if model_args['model'] == 'ResNetCls1':
    #         if model_args['dataset'] == 'cifar10':
    #             # C,H,W = 3,32,32
    #             classifier = ResNetCls1(3, zdim=model_args['latent_dim']).to(device)
    #         elif model_args['dataset'].startswith('celeba'):
    #             # C,H,W = 3,64,64
    #             classifier = ResNetCls1(3, zdim=model_args['latent_dim'], imagesize=64,nclass=1000,resnetl=model_args['resnetl'], dropout=model_args['dropout']).to(device)
    #         else:
    #             # C,H,W = 1,32,32
    #             classifier = ResNetCls1(1, zdim=model_args['latent_dim']).to(device)
    #     elif model_args['model'] == 'vgg':
    #         if model_args['dataset'].startswith('celeba'):
    #             # C,H,W = 3,64,64
    #             classifier =  VGG(zdim=model_args['latent_dim'], nclass=1000, dropout=model_args['dropout']).to(device)
    classifier = get_model(SimpleNamespace(**model_args), device)[0]

    # else: # OLD MODELS...
    #     if dataset == 'mnist':
    #       # C,H,W = 1,model_args.imageSize,model_args.imageSize
    #       classifier = Net(nc=1, nz=128).to(device)
    #     elif dataset in ['cifar10','cifar0to4', 'svhn']:
    #       # C,H,W = 3,model_args.imageSize,model_args.imageSize
    #       classifier = ResNetCls().to(device)
    #     else:
    #       raise ValueError()

    classifier.load_state_dict(torch.load(cls_path))
    classifier.eval()
    return classifier, model_args


def load_cls_z_to_lsm(dataset, cls_path, device):
    classifier, model_args = _load_cls(dataset, cls_path, device)
    return lambda z: classifier.z_to_lsm(z)


def load_cls_embed(dataset, cls_path, device, classify=False, logits=False):
    classifier, model_args = _load_cls(dataset, cls_path, device)
    print(classifier)
    # Which output head
    assert not (classify and logits)
    if classify:
        func = classifier
    elif logits:
        func = classifier.logits
    else:
        func = classifier.embed_img

    H = W = 64  # model_args['imageSize']
    if dataset in ['mnist']:
        C = 1
    elif dataset in ['cifar10', 'cifar0to4', 'svhn']:
        C = 3
    elif dataset.startswith('celeba') or dataset in ['pubfig83', 'cfw']:
        C = 3

    if dataset == 'chestxray':
        C = 1
        H = W = 128

    if dataset == 'mnist':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            return func((x.view(x.size(0), C, H, W) - 0.1307) / 0.3081)
    elif dataset == 'cifar10':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            x = x.view(x.size(0), C, H, W).clone()
            x[:, 0].add_(-0.4914).mul_(1 / 0.2023)
            x[:, 1].add_(-0.4822).mul_(1 / 0.1994)
            x[:, 2].add_(-0.4465).mul_(1 / 0.2010)
            return func(x)
    elif dataset == 'cifar0to4':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            x = x.view(x.size(0), C, H, W).clone()
            x[:, 0].add_(-0.4907).mul_(1 / 0.2454)
            x[:, 1].add_(-0.4856).mul_(1 / 0.2415)
            x[:, 2].add_(-0.4509).mul_(1 / 0.2620)
            return func(x)
    elif dataset == 'svhn':
        def extract_feat(x):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            return func((x.view(x.size(0), C, H, W) - .5) / 0.5)
    elif dataset.startswith('celeba') or dataset in ['pubfig83', 'cfw']:
        def extract_feat(x, mb=100):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            zs = []
            for start in range(0, len(x), mb):
                _x = x[start:start + mb]
                zs.append(func((_x.view(_x.size(0), C, H, W) - .5) / 0.5))
            return torch.cat(zs)
    elif dataset == 'chestxray':
        def extract_feat(x, mb=100):
            assert x.min() >= 0  # in case passing in x in [-1,1] by accident
            zs = []
            for start in range(0, len(x), mb):
                _x = x[start:start + mb]
                zs.append(func((_x.view(_x.size(0), C, H, W) - .5) / 0.5))
            return torch.cat(zs)

    return extract_feat


def parse_kwargs(s):
    r = {}
    if s == '':
        return r
    for item in s.split(','):
        k, tmp = item.split(":")
        t, v = tmp.split(".")
        if t == 'i':
            v = int(v)
        elif t == 'f':
            v = float(v)
        elif t == 's':
            pass
        else:
            raise
        r[k] = v
    return r


def instantiate_generator(args, device):
    if args.gen in ['basic', 'conditional', 'conditional_no_embed']:
        generator = nets.ConditionalGenerator(args.imageSize, args.nz, args.ngf, args.nc, args.n_conditions, args.gen in ['conditional', 'conditional_no_embed'], args.g_sn, args.g_z_scale, args.g_conditioning_method, args.gen != 'conditional_no_embed',
                                              norm=args.g_norm, cdim=args.cdim).to(device)
    elif args.gen in ['secret', 'secret-conditional']:
        generator = nets.ConditionalGeneratorSecret(args.imageSize, args.nz, args.ngf, args.nc, args.n_conditions, args.gen == 'secret-conditional', args.g_sn, args.g_z_scale, args.g_conditioning_method, args.gen != 'conditional_no_embed',
                                                    norm=args.g_norm, cdim=args.cdim).to(device)
    elif args.gen == 'toy':
        generator = nets.ConditionalGeneratorToy(args.imageSize, args.nz, args.ngf, args.nc, args.n_conditions,
                                                 args.use_labels, args.g_sn, args.g_z_scale, args.g_conditioning_method).to(device)
    else:
        raise ValueError()
    return generator


def instantiate_discriminator(args, index2class, device):
    disc_config = yaml.load(open(f'/h/wangkuan/projects/boosted-implicit-models/disc_config/{args.disc_config}', 'r'))
    disc_config['kwargs']['imgSize'] = args.imageSize
    disc_config['kwargs']['nc'] = args.nc
    disc_config['kwargs']['cdim'] = args.cdim
    # Override kwargs
    override_kwargs = parse_kwargs(args.disc_kwargs)
    for k in override_kwargs:
        disc_config['kwargs'][k] = override_kwargs[k]
    if args.use_labels:
        assert disc_config['kwargs']['is_conditional']
        assert args.n_conditions == disc_config['kwargs']['n_conditions']
        disc_config['kwargs']['index2class'] = index2class
    # import ipdb; ipdb.set_trace()
    discriminator = eval(
        'nets.'+disc_config['name'])(**disc_config['kwargs']).to(device)
    print(discriminator)
    print(dir(discriminator))
    return discriminator
