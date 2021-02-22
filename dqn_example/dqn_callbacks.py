#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np

from ray.rllib.agents.callbacks import DefaultCallbacks


class DQNCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
        episode.user_data["heading_deviation"] = []

    def on_episode_step(self, worker, base_env, episode, **kwargs):
        heading_deviation = worker.env.experiment.last_heading_deviation
        if heading_deviation > 0:
            episode.user_data["heading_deviation"].append(heading_deviation)

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        heading_deviation = episode.user_data["heading_deviation"]
        if len(heading_deviation) > 0:
            heading_deviation = np.mean(episode.user_data["heading_deviation"])
        else:
            heading_deviation = 0
        episode.custom_metrics["heading_deviation"] = heading_deviation
