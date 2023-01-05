# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
import os
import uuid
from pathlib import Path

import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for adaptation")
    parser.add_argument("sweep", type=str, help="name of a sweep.")
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--resume", default=None, type=str, help="resume a checkpoint.")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnlab", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")

    args = parser.parse_args()
    return args


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/adaclip")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.args.config.dist_url = get_init_file().as_uri()

    def __call__(self):
        self._setup_gpu_args()
        import main
        main.main(self.args.config)

    def checkpoint(self):
        import os
        import submitit

        self.args.config.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.config.output_dir, "checkpoint-last.pth")
        if os.path.exists(checkpoint_file):
            self.args.config.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        import os
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        if self.args.ngpus >= 1:
            # self.args.config.seed += job_env.global_rank
            # assert 'SLURM_PROCID' in os.environ:
            self.args.config.local_rank = job_env.local_rank
            self.args.config.rank = job_env.global_rank
            self.args.config.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main(args):
    if args.job_dir == "":
        args.job_dir = get_shared_folder()

    assert args.job_dir != ""

    args.job_dir = Path(args.job_dir) / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb= 160 * num_gpus_per_node,  # if "yfcccc12m" not in args.config.output_dir else 120 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=7,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=os.path.basename(args.config.output_dir))
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id, "@", str(args.job_dir).replace("%j", job.job_id))


def submit():
    args = parse_args()
    import sweeps
    import run_configs
    import configs

    from copy import deepcopy
    if hasattr(sweeps, args.sweep):
        print(f"sweeping {args.sweep} in `sweeps.py`")
        sweep_config = getattr(sweeps, args.sweep)
        all_update_dicts = configs.build_from_sweep_config(sweep_config)
        for update_dict in all_update_dicts:
            _args = deepcopy(args)
            config = configs.Config(**update_dict)
            if args.resume is not None:
                config.resume = args.resume
            setattr(_args, "config", config)

            if hasattr(config, "ngpus"):
                _args.ngpus = config.ngpus

            if hasattr(config, "nodes"):
                _args.nodes = config.nodes

            _args.job_dir = config.output_dir
            main(_args)
    elif hasattr(run_configs, args.sweep):
        print(f"launch {args.sweep} in `run_configs.py`")
        config = getattr(run_configs, args.sweep)()
        _args = deepcopy(args)
        if args.resume is not None:
            config.resume = args.resume
        setattr(_args, "config", config)

        if hasattr(config, "ngpus"):
            _args.ngpus = config.ngpus

        if hasattr(config, "nodes"):
            _args.nodes = config.nodes

        _args.job_dir = config.output_dir
        main(_args)


if __name__ == "__main__":
    submit()
