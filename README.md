# CARLA and RLlib integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLlib integration brings support between the [Ray/RLlib](https://github.com/ray-project/ray) library and the [CARLA simulator](https://github.com/carla-simulator/carla), allowing the easy use of the CARLA environment for training and inference purposes.

## Project Organization

This repository is organized as follows:

* **rllib_integration** contains all the infraestructure used to set up the CARLA server, clients and the training and testing experiments. Additionally, if you want to create your own pseudosensor, check out **sensors/birdview_manager.py**, which is a simplified version of [CARLA's no rendering mode](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/no_rendering_mode.py).

* **aws** has the files needed to run this in an AWS instance. Specifically, the **aws_helper.py** provides several functionalities that ease the management of the EC2 instances, including their creation as well as retrieving and sending data. Check the next section to show how to use it

* **dqn_example**, as well as all the **dqn_*** files provide an easy to understand example on how to use the tools available at the previously explained folder.

## Running the repository locally

In order to run this RLlib integration, the following steps have to be taken:

### Setting up CARLA and dependencies

As CARLA is the environment that ray will be using, the first step is to set it up. To do so, **a packaged version will have to be installed** (see all [**CARLA releases**](https://github.com/carla-simulator/carla/releases)). This integration has been done using CARLA 0.9.11 and therefore it is recommended to use that version. While other versions might be compatible, they haven't been fully tested, so procede at your own discretion.

Additionally, in order to know where this package is located, set the **CARLA_ROOT** environment variable to the path of the folder.

With CARLA installed, we can install the rest of the prerequisites with:

`pip3 install -r requirements.txt`

### Running the training

With all the setup complete, the only step missing is to run the training script:

`python3 <training_file> <training_config>`

## Running on AWS

Additionally, we also provide tools to automatically run the training on EC2 instances. To that end, we make use of the [**Ray autoscaler**](https://docs.ray.io/en/latest/cluster/index.html) API.

### Creating the training AMI

The first step is to create the image needed for training. We provide a script that automatically creates it, using the Deep Learning AMI (Ubuntu 18.04) as the base image:

```
python3 aws_helper.py create-image --name <AMI-name> --installation-scripts <installation-scripts> --instance-type <instance-type> --volume-size <volume-size>
```

**Note:** This script will end by outputting information about the created image. In order to use RAY's autoscaler, manually update the image id and security group id information at your autoscaler configuration file with the provided ones.

### Running the training AMI

With the image created, we can use RAY's API to run the training at the cluster:

1. Initialize the cluster:

```
ray up <autoscaler_configuration_file>
```

2. (Optional) Update remote files with local changes:

If the local code has been modified after the cluster initialization, use these command lines to update it.

```
ray rsync-up <path_to_local_folder> <path_to_remote_folder>
```

3. Run the training:

```
ray submit <autoscaler_configuration_file> <training_file>
```

4. (Optional) Monitor the cluster status:

```
ray attach <autoscaler_configuration_file>
watch -n 1 ray status
```

5. Shutdown the cluster:

```
ray down <autoscaler_configuration_file>
```

## Breakdown of the example experiment

This next section dives deeper into all the files used by the DQN example. For this example, ray's [DQNTrainer](https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py#L285) is used, and both training and inference, are showcased. The files are as follows:

* **dqn_train.py** is the entry to the training. It has one compulsory argument, which should be the path to the experiment configuration. Here, the configuration is parsed and given to ray, along with the experiment and CARLA environment classes.
* **dqn_example/dqn_config.yaml** is a yaml configuration file. This should be the argument given to the _dqn_train.py_ file. Both the experiment and CARLA settings, as well as the ray ones can be changed from here. These will be updated on top of the default ones.
* **dqn_inference.py** is the script used to start an inference. For this example, one argument is needed, the path to a pytorch checkpoint. 
* **dqn_inference_ray.py** holds the same functionality as the previous inference file but in this case, it is done using the 
ray library. As with the previous file, it also requires a cehckpoint file, which should be automatically be created by the trainer.
* At **dqn_example/dqn_experiment.py**, the experiment class is created, with all the information regarding the action space, observation space, actions and rewards needed for the DQN. It is recommended that all experiments inherit from `BaseExperiment`, at **rllib_integration/base_experiment.py**.

With its structure explain, the next sections will be used to showcase how to run this specific example.

### Running the example locally

As explained before, to start the training, the only command you need is to run a python file. In this case:

```
python3 dqn_train.py dqn_example/dqn_config.yaml --name dqn
```

**Note:** The default configuration uses 1 GPU and 12 CPUs, so if your current instance doesn't have that amount of capacity, lower the numbers at the `dqn_example/dqn_config.yaml`.

### Running the example on AWS

On the other hand, if you want to run this example on AWS, you have to first create the AMI with:

```
python3 aws_helper.py create-image --name <AMI-name> --installation-scripts install/install.sh --instance-type <instance-type> --volume-size <volume-size>
```

**Note:** Remember to change the image id and security group id at [`dqn_example/dqn_autoscaler.yaml`](https://github.com/carla-simulator/rllib-integration/blob/readme/dqn_example/dqn_autoscaler.yaml#L39).

And then, initialize the clusters and start the training:

```
# Initialize the cluster
ray up dqn_example/dqn_autoscaler.yaml

# (Optional) Update remote files with local changes
ray rsync-up dqn_example/dqn_autoscaler.yaml dqn_example .
ray rsync-up dqn_example/dqn_autoscaler.yaml rllib_integration .

# Run the training
ray submit dqn_example/dqn_autoscaler.yaml dqn_train.py -- dqn_example/dqn_config.yaml --auto

# (Optional) Monitor the cluster status 
ray attach dqn_example/dqn_autoscaler.yaml
watch -n 1 ray status

# Shutdown the cluster
ray down dqn_example/dqn_autoscaler.yaml
```
