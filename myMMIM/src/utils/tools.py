import torch
import os
import io


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model):
    pretrained_path = args.save_dir + r'/pre_trained_models'
    if not os.path.exists( pretrained_path ):
        os.makedirs( pretrained_path )
    torch.save(model.state_dict(), pretrained_path + f'/{args.model_name}.pt')


def load_model(args, model):
    pretrained_path = args.save_dir + r'/pre_trained_models'
    model.load_state_dict(torch.load( pretrained_path + f'/{args.model_name}.pt'))
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)
    
    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)
    
    return t
