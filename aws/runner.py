#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os

import aws.utils as utils


class AWSRunner(object):
    def __init__(self, instance_id):
        self.instance_id = instance_id

    def run(self, script, args="", rsync_folder=True):
        instance = utils.get_instance(self.instance_id)
        if instance is None:
            raise RuntimeError("instance id %s not found", self.instace_id)
        if instance.state["Name"] == "stopped":
            utils.start_instance(instance)

        if rsync_folder:
            utils.put(instance, os.path.dirname(script))
        utils.exec_script(instance, script, args)

    def stop(self):
        instance = utils.get_instance(self.instance_id)
        utils.stop_instance(instance)