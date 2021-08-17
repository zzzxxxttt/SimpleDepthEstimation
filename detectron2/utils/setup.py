import os
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.layers.fakeDDP import FakeDDP
from detectron2.checkpoint import DetectionCheckpointer


def simple_main(args, add_cfg_fn, train_fn, test_fn=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cfg_fn(cfg)
    cfg.merge_from_file(args.config_file)
    run_name = os.path.splitext(args.config_file.split('/')[-1])[0] + ('_' + args.etag if args.etag != '' else '')
    cfg.merge_from_list(args.opts + ["RUN_NAME", run_name])
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.RUN_NAME)
    cfg.freeze()
    default_setup(cfg, args)  # if you don't like any of the default setup, write your own setup code

    model = build_model(cfg)
    # logger.info("Model:\n{}".format(model)) # note: uncomment this to see the model structure
    if args.eval_only and test_fn is not None:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return test_fn(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[comm.get_local_rank()],
                                        broadcast_buffers=False,
                                        find_unused_parameters=True)
    else:
        model = FakeDDP(model)

    train_fn(cfg, model, resume=args.resume)
    if test_fn is not None:
        return test_fn(cfg, model)
    else:
        return
