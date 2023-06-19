import argparse
import yaml
import torch.multiprocessing as mp
import torch.distributed as dist


# NOTE: This currently only runs on a single node with multiple GPUs
# I can extend this if necessary

# This code is NOT for running directly. Create a new function based on the example below using setup and cleanup, then
# call ddp_main(function, function_args) in the if __name__ == '__main__' block


def setup(gpu, args):
    rank = gpu  # + args.nr * args.gpus
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


def function(gpu, args):
    setup(gpu, args)

    # program code goes here
    # don't forget to wrap your model in DDP!
    print(f"This is an example function running on rank {gpu}")

    cleanup()

    return


def ddp_main(function):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-c', '--config', default='configs/ps.yml', type=str,
                        help='config file path (default: configs/ps.yml)')
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    args.config = config

    args.world_size = args.gpus * args.nodes
    mp.spawn(function, nprocs=args.gpus, args=(args,))



