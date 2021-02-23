# CARLA and RLlib integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLlib integration brings support between the [Ray/RLlib](https://github.com/ray-project/ray) library and the [CARLA simulator](https://github.com/carla-simulator/carla). This repository handles the creation and use of the CARLA simulator as an environment of Ray, which the users can use for training and inference purposes. This is complemented by an example, as well as some files to ease the use of AWS instances. These functionalities are divided in the following way:

* **rllib_integration** contains all the infrastructure related to CARLA. Here, we set up the CARLA server, clients and actors. Also, the basic structure that all training and testing experiments must follow is included here.

* **aws** has the files needed to run this in an AWS instance. Specifically, the **aws_helper.py** provides several functionalities that ease the management of the EC2 instances, including their creation as well as retrieving and sending data.

* **dqn_example**, as well as all the others **dqn_*** files, provide an easy-to-understand example on how to set up a Ray experiment using CARLA as its environment.

## Setting up CARLA and dependencies

As CARLA is the environment that Ray will be using, the first step is to set it up. To do so, **a packaged version must be installed** (see all [**CARLA releases**](https://github.com/carla-simulator/carla/releases)). This integration has been done using CARLA 0.9.11, and therefore it is recommended to use that version. While other versions might be compatible, they haven't been fully tested, so proceed at your own discretion.

Additionally, in order to know where this package is located, set the **CARLA_ROOT** environment variable to the path of the folder.

**Note**: It is only needed to install CARLA if you want to run this repository locally.

With CARLA installed, we can install the rest of the prerequisites with:

`pip3 install -r requirements.txt`


## Creating your own experiment

Let's start by explaining how to create your own experiment. To do so, you'll need to create:

- Experiment class
- Configuration file
- Training and inference files


### Create the experiment class

The first step that you need to do is to define a training experiment. For all environments to work with Ray, they have to return specific information (see [**CarlaEnv**](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/carla_env.py)), which will be dependent on your chosen experiment. As such, all experiments should inherit from [**BaseExperiment**](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/base_experiment.py#L39), overwritting all of its methods.

### Configure the environment 

Additionally, a configuration file is also required. Any settings here update the default ones. It can be divided in three parts:

- **Ray trainer configuration**: everything related to the specific trainer used. If you are using a built-in model, you can set up its settings here.
- **CARLA environment**: CARLA related settings. These can be divided into the simulation, such as timeout or map quality (default values [here](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/carla_core.py#L23)); and the experiment configuration, related to the ego vehicle and its sensors (check default settings for how to specificy the sensors to use), as well as the town conditions (default values [here](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/base_experiment.py#L12)).

### Create the training and inference files

The last step is to create your own training and inference files. This part is completely up to you and is dependent on the Ray API. Remember to check [Ray's custom model docs](https://docs.ray.io/en/master/rllib-models.html#custom-models-implementing-your-own-forward-logic), if you want to create your own specific model.


## DQN example

To solidify the previous section, we also provide a simple example. It uses the [BirdView pseudosensor](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/sensors/bird_view_manager.py), along with Ray's [DQNTrainer](https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py#L285). The files are:

- The [**DQNExperiment**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py#L19) is the experiment class, which overwrites the methods of its parent class.
- The configuration file is [**dqn_example/dqn_config.yaml**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_config.yaml).
- [**dqn_train.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_train.py) is responsible for the training, [**dqn_inference_ray.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_inference_ray.py) of the inference using Ray's API, and [**dqn_inference.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_inference.py), of the inference without it.

To run this example locally, you need to install pytorch
```bash 
pip3 install -r dqn_example/dqn_requirements.txt
```
and run the training file
```bash
python3 dqn_train.py dqn_example/dqn_config.yaml --name dqn
```

**Note:** The default configuration uses 1 GPU and 12 CPUs, so if your current instance doesn't have that amount of capacity, lower the numbers at the `dqn_example/dqn_config.yaml`. Additionally, if you are having out of memory problems, consider reducing the `buffer_size` parameter.


## Running on AWS

Additionally, we also provide tools to automatically run the training on EC2 instances. To do so, we use the [**Ray autoscaler**](https://docs.ray.io/en/latest/cluster/index.html) API.

### Configure AWS

Firstly, configure your boto3 environment correctly. You can follow the instructions [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html).

### Creating the training AMI

The first step is to create the image needed for training. We provide a script that automatically creates it, given the base image and the installation script:

```
python3 aws_helper.py create-image --name <AMI-name> --installation-scripts <installation-scripts> --instance-type <instance-type> --volume-size <volume-size>
```

**Note:** This script will end by outputting information about the created image. In order to use Ray autoscaler, manually update the image id and security group id information at your autoscaler configuration file with the provided ones.

### Running the training

With the image created, we can use Ray's API to run the training at the cluster:

1. Initialize the cluster:

```
ray up <autoscaler_configuration_file>
```

2. (Optional) Update remote files with local changes:

If the local code has been modified after the cluster initialization, use these command lines to update it.

```
ray rsync-up <autoscaler_configuration_file> <path_to_local_folder> <path_to_remote_folder>
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

### Running the DQN example on AWS

For this example, we use the autoscaler configuration at [dqn_example/dqn_autoscaler.yaml](https://github.com/carla-simulator/rllib-integration/blob/readme/dqn_example/dqn_autoscaler.yaml#L39). To execute it, you just need to run:

```bash
# Create the training image 
python3 aws_helper.py create-image --name <AMI-name> --installation-scripts install/install.sh --instance-type <instance-type> --volume-size <volume-size>
```

**Note**: Remember to manually change the image id and security group id at the `dqn_example/dqn_autoscaler.yaml` after this command line

```bash
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

