#!/usr/bin/env python3
from absl import app
from absl import flags

from lxm3 import xm
from lxm3 import xm_cluster
from lxm3.contrib import ucl

_LAUNCH_ON_CLUSTER = flags.DEFINE_boolean(
    "launch_on_cluster", False, "Launch on cluster"
)
_GPU = flags.DEFINE_boolean("gpu", False, "If set, use GPU")
_SINGULARITY_CONTAINER = flags.DEFINE_string(
    "container", "svf_gymnasium-latest.sif", "Path to singularity container"
)


def main(_):
    with xm_cluster.create_experiment(experiment_title="basic") as experiment:
        # It's actually not necessary to use a container, without it, we
        # fallback to the current python environment for local executor and
        # whatever Python environment picked up by the cluster for GridEngine.
        # For remote execution, using the host environment is not recommended.
        # as you may spend quite some time figuring out dependency problems than
        # writing a simple Dockfiler/Singularity file.
        singularity_container = _SINGULARITY_CONTAINER.value
        if _GPU.value:
            job_requirements = xm_cluster.JobRequirements(
                gpu=1,
                ram=16 * xm.GB,
                cpu=16,
            )
        else:
            job_requirements = xm_cluster.JobRequirements(ram=16 * xm.GB, cpu=16)
        print(job_requirements.task_requirements)
        if _LAUNCH_ON_CLUSTER.value:
            # This is a special case for using SGE in UCL where we use generic
            # job requirements and translate to SGE specific requirements.
            # Non-UCL users, use `xm_cluster.GridEngine directly`.
            executor: xm_cluster.GridEngine = ucl.UclGridEngine(
                job_requirements,
                walltime=4 * xm.Hr,
            )
        else:
            executor = xm_cluster.Local(job_requirements)

        spec = xm_cluster.PythonPackage(
            # This is a relative path to the launcher that contains
            # your python package (i.e. the directory that contains pyproject.toml)
            path="..",
            # Entrypoint is the python module that you would like to
            # In the implementation, this is translated to
            #   python3 -m py_package.main
            entrypoint=xm_cluster.ModuleName("svf_gymnasium.train"),
        )

        # Wrap the python_package to be executing in a singularity container.
        if singularity_container is not None:
            spec = xm_cluster.SingularityContainer(
                spec,
                image_path=singularity_container,
            )

        [executable] = experiment.package(
            [xm.Packageable(spec, executor_spec=executor.Spec())]
        )

        # To submit parameter sweep by array jobs, you can use the batch context
        # Without the batch context, jobs with be submitted individually.
        with experiment.batch():

            global_args = {
                "algo": "sac",
                "env": "Hopper-v4",
                "conf": "svf_gymnasium/hyperparams/sac.yaml",
                "track": True,
                "wandb-entity": "dtch1997",
                "wandb-project": "svf_gymnasium",
            }

            envs = [
                "Hopper-v4",
                "Safe-Hopper-v4",
                "HalfCheetah-v4",
                "Safe-HalfCheetah-v4",
            ]

            num_envs = len(envs)
            for i in range(2 * len(envs)):

                args = global_args.copy()
                # Configure run-specific args
                if i // num_envs == 0:
                    args["env"] = envs[i]
                elif i // num_envs == 1:
                    args["env"] = f"Safe-{envs[i % num_envs]}"

                experiment.add(
                    xm.Job(
                        executable=executable,
                        executor=executor,
                        # You can pass additional arguments to your executable with args
                        # This will be translated to `--seed 1`
                        # Note for booleans we currently use the absl.flags convention
                        # so {'gpu': False} will be translated to `--nogpu`
                        args=args,
                        # You can customize environment_variables as well.
                        # env_vars={"TASK": str(i)},
                    )
                )

        # You can also use a job generator.
        # This is useful for example in a few cases
        # 1. if you want to configure a working directory depending on the work unit id.
        # 2. You can to dynamically compute some additional args/env_vars.
        #    one use case is to compute environment variables to be passed to wandb.
        # async def make_job(work_unit: xm.WorkUnit, **args) -> None:
        #     work_unit.add(
        #         xm.Job(
        #             executable=executable,
        #             executor=executor,
        #             args={"seed": args["i"]},
        #             env_vars={"TASK": work_unit.work_unit_id},
        #         )
        #     )

        # with experiment.batch():
        #     for i in range(2):
        #         experiment.add(make_job, {"i": i})


if __name__ == "__main__":
    app.run(main)
