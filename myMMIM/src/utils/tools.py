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
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    torch.save(model.state_dict(), f'pre_trained_models/{args.model_name}.pt')


def load_model(args, model):
    model.load_state_dict(torch.load(f'pre_trained_models/{args.model_name}.pt'))

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
