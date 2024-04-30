import os
import json
import argparse
import copy
import numpy as np

import torch

from timm import create_model

from util.datasets import build_dataset, build_calib_loader
from util.profiler import torch_profiler, onnx_profiler
from util.misc import print_table
from util.load import make_pruned_model
from engine import evaluate

from onnx_quant import ONNX_inference


def get_args_parser():
    parser = argparse.ArgumentParser(description='ONNX_PTQ', add_help=False)

    parser.add_argument('--device', default='cuda', help='cpu vs cuda')

    #data load
    #/mnt/d/data/image/ILSVRC/Data/CLS-LOC
    parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'])
    parser.add_argument('--data_path', default='/home/kimtaeho', type=str, help='path to ImageNet data')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    #model
    parser.add_argument('--model', default="vgg16", type=str, help='model name',
                        choices=['vgg16', 'resnet18', 'resnet34', 'resnet50', 'resnet54'])
    parser.add_argument('--pruned_pretrained', default='./res18_pruned_checkpoint.pth', help='get pretrained weights from checkpoint')
    parser.add_argument('--ori_pretrained', default='./res18_best_checkpoint.pth', help='get pretrained weights from checkpoint')

    #calibration
    parser.add_argument('--num_samples', default=1000, type=int, help='size of the calibration dataset')
    parser.add_argument('--calib_batch_size', default=10, type=int, help='number of iterations for calibration')

    #save
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save scale, empty for no saving')


    parser.add_argument('--print_freq', default=500, type=int)

    return parser

def main(args):

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.output_dir:
        output_dir = args.output_dir + f"/{args.model}"
        os.makedirs(output_dir, exist_ok=True)


    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    #get calibraition dataloader
    calib_loader = build_calib_loader(dataset_train, num_samples=args.num_samples, seed=seed, args=args)

    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=10, 
        num_workers=3,
        drop_last=False
    )

    # load model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
    ) 
    ori_model = copy.deepcopy(model)

    if args.ori_pretrained:
        if args.ori_pretrained.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.ori_pretrained, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.ori_pretrained, map_location='cpu')

        msg = ori_model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Original model load checkpoint {msg}")
          
    if args.pruned_pretrained:
        checkpoint = torch.load(args.pruned_pretrained, map_location='cpu')
        if "save_size" in checkpoint.keys():
            save_size = checkpoint["save_size"]
            print(f"-"*50 + "Resuming" + "-"*50)
            print(f"building pruned model using checkpoint['save_size']")
            model = make_pruned_model(model, "", save_size)
        else:
            raise ValueError("checkpoint does not have save_size key")

        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Pruned model load checkpoint {msg}")

    onnx_stats, onnx_path = ONNX_inference(calib_loader, model, data_loader_val, output_dir, args)
    onnx_stats["name"] = f"onnx_{args.model}_compressed"
    dummy_size = (1, 3, args.input_size, args.input_size)
    onnx_pf = onnx_profiler(dummy_size=dummy_size)
    onnx_summary = onnx_pf.summary(onnx_path)
    onnx_stats.update(onnx_summary)

    if args.ori_pretrained:
        torch_stats = evaluate(data_loader_val, ori_model, "cpu", args)
        torch_pf = torch_profiler(dummy_size=dummy_size)
        torch_stats["name"] = f"torch_{args.model}_ori"
        torch_stats["latency(ms)"] = torch_pf.torch_model_latency(ori_model)
        torch_stats["size(mb)"] = torch_pf.torch_model_size(ori_model)

    torch_pruned_stats = evaluate(data_loader_val, model, "cpu", args)
    torch_pruned_stats["name"] = f"torch_{args.model}_pruned"
    torch_pruned_stats["latency(ms)"] = torch_pf.torch_model_latency(model)
    torch_pruned_stats["size(mb)"] = torch_pf.torch_model_size(model)



    data = [onnx_stats, torch_pruned_stats, torch_stats]
    print_table(data)

    
    with open(f"{output_dir}/compression_summary.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    args = get_args_parser()    
    args = args.parse_args()
    main(args)