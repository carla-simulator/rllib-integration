# CARLA and RLlib integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLlib integration brings support between the [Ray/RLlib](https://github.com/ray-project/ray) library and the [CARLA simulator](https://github.com/carla-simulator/carla). This repository handles the creation and use of the CARLA simulator as an environment of RAY, which the users can use for training and inference purposes. This is complemented by an example, as well as some files to easy the use of AWS instances. These functionalities are divided in the following way:

* **rllib_integration** contains all the infrastructure related to CARLA. Here, we set up the CARLA server, clients and actors. Also, the basic structure that all training and testing experiments must follow is included here.

<!-- Additionally, if you want to create your own pseudo-sensor, check out **sensors/birdview_manager.py**, which is a simplified version of [CARLA's no rendering mode](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/no_rendering_mode.py). -->

* **aws** has the files needed to run this in an AWS instance. Specifically, the **aws_helper.py** provides several functionalities that ease the management of the EC2 instances, including their creation as well as retrieving and sending data.

* **dqn_example**, as well as all the others **dqn_*** files, provide an easy-to-understand example on how to set up a RAY experiment using CARLA as its environment.





## Creating your own experiment

Let's start by explaining how to create your own experiment. To do so, you'll need to create at least three files. These are the experiment class, the training file and the configuration of the environment. While this section focuses on a general overview of what you need to so, the next one explains an specific example, so make sure to check both of them for a better understanding.

### Create the experiment class

The first step that you need to do to use the CARLA environment is to define a training experiment. For all environments to work with RAY, they have to return a series of specific information (see [**CarlaEnv**](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/carla_env.py)), which will be dependent on your specific experiment. As such, all experiments should inherit from [**BaseExperiment**](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/base_experiment.py#L39), where all the functions that need to be overwritten are located. These are all functions related to the actions, observations and rewards of the training experiment.

### Configure the environment 

Apart your experiment, a configuration file is also required. Any settings here update the default ones. Its purpose is threefold. Firstly, you can set up most of the CARLA server and client settings directly from this file (see the default values [here](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/carla_core.py#L23)). Secondly, and similarly to the previous point, your experiment also has a default configuration (see [here](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/base_experiment.py#L12)), which handles the spawning of the ego vehicle and its sensors (check the default settings for how to specificy the sensors to use), as well as the town conditions. Any other specific variables of your experiment can also be placed here. The rest of the settings are related to RAY's training.

### Create the training file

The last step is to create your own training file. This part is completely up to the user and is dependent on the RAY API.





## Running the repository locally

In order to run this RLlib integration, the following steps have to be taken:

### Setting up CARLA and dependencies

As CARLA is the environment that Ray will be using, the first step is to set it up. To do so, **a packaged version must be installed** (see all [**CARLA releases**](https://github.com/carla-simulator/carla/releases)). This integration has been done using CARLA 0.9.11, and therefore it is recommended to use that version. While other versions might be compatible, they haven't been fully tested, so proceed at your own discretion.

Additionally, in order to know where this package is located, set the **CARLA_ROOT** environment variable to the path of the folder.

With CARLA installed, we can install the rest of the prerequisites with:

`pip3 install -r requirements.txt`

### Running the training

With all the setup complete, the only step missing is to run the training script:

`python3 <training_file> <training_config>`





## Running on AWS

Additionally, we also provide tools to automatically run the training on EC2 instances. To do so, we use the [**Ray autoscaler**](https://docs.ray.io/en/latest/cluster/index.html) API.

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



## DQN example

Due to the verstaility of both CARLA and RAY, we also provide an example on how to use this integration, which will help solidify what we talked about at the previous section, diving deeper into all the files used by the example. This example is based on RAY's [DQNTrainer](https://github.com/ray-project/ray/blob/master/rllib/agents/dqn/dqn.py#L285).

### Creation of the experiment

As previously explained, the first step is to create the experiment class. This can be found at [**dqn_example/dqn_experiment.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py). As a general overview, it uses a discrete action space with a length of 28 actions, and the observations sent to RAY are postprocessed images created by the [BirdView Pseudosensor](https://github.com/carla-simulator/rllib-integration/blob/main/rllib_integration/sensors/bird_view_manager.py).

Regarding the configuration, it is located at [**dqn_example/dqn_config.yaml**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_config.yaml). The first set of settings are related to the DQNTrainer, followed by the CARLA environment and the experiment ones. Focus of the latter, the training blueprint of the ego vehicle is fixed to the Lincoln MKZ2017, and we tell CARLA to create the 'birdview' sensor. By default, the ego vehicle is spawned randomly throughout the map but if one or more spawn points are set, it chooses randomly between those. There are also additional settings such as the experiment town, or the amount of background activity roaming the city.

For the training, the [**dqn_train.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_train.py) is used. Here, we use RAY [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html) to run the trainer and periodically save some checkpoints. Additionally, there are two files that, using the aforementioned checkpoints allow you to use inference on the model ([**dqn_inference.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_inference_ray.py) uses RAY's API while [**dqn_inference.py**](https://github.com/carla-simulator/rllib-integration/blob/main/dqn_inference.py) doesn't)

With its structure explain, the next sections will be used to showcase how to run this specific example.

### Running the example locally

As explained before, to start the training, the only command you need is to run a python file. In this case:

```
python3 dqn_train.py dqn_example/dqn_config.yaml --name dqn
```

**Note:** The default configuration uses 1 GPU and 12 CPUs, so if your current instance doesn't have that amount of capacity, lower the numbers at the `dqn_example/dqn_config.yaml`.

### Running the example on AWS

On the other hand, if you want to run this example on AWS, you have to first create the AMI. For this example, we use the Deep Learning AMI (Ubuntu 18.04) as the base image:

```
python3 aws_helper.py create-image --name <AMI-name> --installation-scripts install/install.sh --instance-type <instance-type> --volume-size <volume-size>
```

**Note:** Remember to change the image id and security group id at [`dqn_example/dqn_autoscaler.yaml`](https://github.com/carla-simulator/rllib-integration/blob/readme/dqn_example/dqn_autoscaler.yaml#L39).

And then, initialize the cluster and start the training:

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
