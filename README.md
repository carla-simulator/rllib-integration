[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLlib integration brings support between the [Ray/RLlib](https://github.com/ray-project/ray) library and the [CARLA simulator](https://github.com/carla-simulator/carla), allowing the easy use of the CARLA environment for training and inference purposes.

## Setup Carla

As CARLA is the environment that ray wll be using, the first step is to set it up. To do so, a packaged version will have to be installed (see all [**CARLA releases**](https://github.com/carla-simulator/carla/releases). This integration has been done using CARLA 0.9.11 and therefore it is recommended to use that version. While other versions might be compatible, they haven't been fully tested, so procede at your own discretion. Additionally, in order to know where this packaged is located, set the **CARLA_ROOT** environment variable to the folder containing it.

## Project Organization

This repository is organized as follows:

* **aws** has the files needed to run this in an AWS instance. Specifically, the **aws_helper.py** cprovides several functionalities that ease the management of the EC2 instances, including their creation as well as retrieving and sending data. Check the next section to show how to use it

* **rllib_integration** contains all the infraestructure used to set up the CARLA server, clients and the training and testing experiments. It is important to note that **base_experiment.py** has the base settings of the experiments (at `BASE_EXPERIMENT_CONFIG`), as well as the `BASE_CORE_CONFIG` at **carla_core.py**, where all the CARLA default settings are set up. Additionally, if you want to create your own pseudosensor, check out **sensors/birdview_manager.py**, which is a simplified version of [CARLA's no rendering mode](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/no_rendering_mode.py).

* **dqn_example**, as well as all the **dqn_*** files provide an easy to understand example on how to use the tools available at the previously explained folder.

## Breakdown of the example experiment

This next section dives deeper into all the files used by the DQN example. For this example, ray's [DQNTrainer](https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py#L285) is used, and both training and inference, are showcased. The files are as follows:

* **dqn_train.py** is the entry to the training. It has one compulsory argument, which should be the path to the experiment configuration. Here, the configuration is parsed and given to ray, along with the experiment and CARLA environment classes.
* **dqn_config.yaml** is a yaml configuration file. This should be the argument given to the _dqn_train.py_ file. Both the experiment and CARLA settings, as well as the ray ones can be changed from here. These will be updated on top of the default ones.
* **dqn_inference.py** is the script used to start an inference. For this example, one argument is needed, the path to a pytorch checkpoint. 
* **dqn_inference_ray.py** holds the same functionality as the previous inference file but in this case, it is done using the 
ray library.
* At **dqn_example/dqn_experiment.py**, the experiment class is created, with all the information regarding the action space, observation space, actions and rewards needed for the DQN. It is recommended that all experiments inherit from `BaseExperiment`, at **rllib_integration/base_experiment.py**.

## Running on the cloud

Additionally, we also provide tools used to automatically run the training on an EC2 instance.

### Creation of the training AMI

The first step is to create the image needed for training. We provide a script that automatically creates it, using the Deep Learning AMI (Ubuntu 18.04) as the base image. Additionally, we also install CARLA 0.9.11 and all the needed libraries inside a conda environment. In order to execute create the AMI, run:

`python3 aws_helper.py create-image --installation-scripts aws/install/install.sh --name <AMI-name> --instance-type <instance-type> --volume-size <volume-size> `

**Note:** The recommended volume size is 500, and the instance type, g4dn.12xlarge

### Running the training at the instance

With the image created, we can now run the training at the instance. Run:

`bash run_aws.bash -i <instance-id> -s <training_file> [-c <configuration_file.yaml> -n <experiment-name> -d <directory-name>]`

With this command, the instance will be automatically started (if needed) and the training process will begin. Any argument that is accepted by the training file can also be passed directly to the `run_aws.bash` file.
