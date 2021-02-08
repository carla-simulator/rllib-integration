[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RLlib integration brings support between the [Ray/RLlib](https://github.com/ray-project/ray) library and the [CARLA simulator](https://github.com/carla-simulator/carla), allowing the easy use of the CARLA environment for training and inference purposes.


## Setup Carla

To use CARLA, go to the [**CARLA releases**](https://github.com/carla-simulator/carla/releases) webpage and download version 0.9.11. While other versions might be compatible, they haven't been fully tested, so procede at your own discretion.

Additionally, set the **CARLA_ROOT** environment variable to the folder containing the CARLA server.

## Project Organization

This repository is organized as follows:

* **rllib_integration** contains all the infraestructure used to set up the CARLA server, clients and the training and testing experiments:
    * At the **sensors** folder, all the available sensors are defined. If you want to create your own pseudosensor, check the `BirdviewSensor`, which a simplified version of the [CARLA's no rendering mode](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/no_rendering_mode.py).
    * **base_experiment.py** is the base class from which all experiments should inherit. Here, a series of not implemented functions are shown. This are the ones that will have to created by your experiment in order to ensure its correct usage. Also, the `BASE_EXPERIMENT_CONFIG` dictionary shows the default configuration of the experiments.
    * **carla_core.py** handles all the client-server actions, as well as the spawning of all the actors in the simulation, including the sensors used. `BASE_CORE_CONFIG` is the base configuration of CARLA.
    * **carla_env.py** creates the CARLA environment that will connect to ray.
    * **helper.py** englobes different functions that might be useful.
* **dqn_example**, as well as all the **dqn_*** files provide an easy to understand example on how to use the tools available at the previously explained folder, which uses the default _DQNTrainer_ ray class. This includes both training and inference:
    * **dqn_train.py** is the entry to the training. It has one compulsory argument, which should be the path to the experiment configuration. Here, the configuration is parsed and given to ray, along with the experiment and CARLA environment classes.
    * **dqn_config.yaml** is a yaml configuration file. This should be the argument given to the _dqn_train.py_ file. Both the experiment and CARLA settings, as well as the ray ones can be changed from here. These will be updated on top of the default ones.
    * **dqn_inference.py** is the script used to start an inference. For this example, one argument is needed, the path to a pytorch checkpoint. 
    * **dqn_inference_ray.py** holds the same functionality as the previous inference file but in this case, it is done using the 
    ray library
    * At **dqn_example/dqn_experiment.py**, the experiment class is created, with all the information regarding the action space, observation space, actions and rewards needed for DQN.

* **aws** has the additional files to run all of this in an AWS instance.
