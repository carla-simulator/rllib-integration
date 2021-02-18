#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
cli to ease the management of AWS EC2 instances with the integration of CARLA and RLlib
"""

import argparse
import logging
import sys
import os

import utils

# Deep Learning AMI (Ubuntu 18.04) Version 39.0
DEFAULT_AMI = {
    "us-east-1": "ami-03e0fdb8c9d235984",  # US East (N. Virginia)
    "us-east-2": "ami-0edc3c56e8af8d35a",  # US East (Ohio)
    "us-west-1": "ami-016a12e19041b168e",  # US West (N. California)
    "us-west-2": "ami-0b1a80ce62c464a55",  # US West (Oregon)
    #"af-south-1": "",  # Africa (Cape Down)
    #"ap-east-1": "",  # Asia Pacific (Hong kong)
    "ap-south-1": "ami-0d9863d058a268c58",  # Asia Pacific (Mumbai)
    "ap-northeast-2": "ami-05f907e64fc9e2e78",  # Asia Pacific (Seoul)
    "ap-southeast-1": "ami-0b96a6648d6e7935d",  # Asia Pacific (Singapore)
    "ap-southeast-2": "ami-035893f193fa7bf5d",  # Asia Pacific (Sidney)
    "ap-northeast-1": "ami-0110696785c0c8ee2",  # Asia Pacific (Tokyo)
    "ca-central-1": "ami-0c374202c028e797a",  # Canada (Central)
    "eu-central-1": "ami-001ec343bb21e7e59",  # Europe (Frankfurt)
    "eu-west-1": "ami-0e5d3cb86ff6f2dcb",  # Europe (Ireland)
    "eu-west-2": "ami-05f287799f4865acb",  # Europe (London)
    #"eu-south-1": "",  # Europe (Milan)
    "eu-west-3": "ami-0d84b78b184f013ed",  # Europe (Paris)
    #"eu-north-1": "",  # Europe (Stockholm)
    #"me-south-1": "",  # Middle East (Bahrain)
    "sa-east-1": "ami-017cd2fe7463e6ac1"  # South America (Sao Paulo)
}


def create_ami(args):
    logging.info(
        """Creating AMI...
        name: %s
        base ami id: %s
        instance type: %s
        key name: %s
        security group: %s
        volume size: %s
        installation scripts: %s
        user data: %s""", args.name, args.base_ami_id, args.instance_type, args.key_name,
        args.security_group_name, args.volume_size, args.installation_scripts, args.user_data)
    instance, image = utils.create_image(args.name, args.base_ami_id,
                                         args.instance_type, args.key_name,
                                         args.security_group_name,
                                         args.volume_size,
                                         args.installation_scripts,
                                         args.user_data)

    print("\n")
    utils.print_image_info(image)
    print("\n")
    utils.print_instance_info(instance)

def launch(args):
    logging.info(
        """Launching EC2 instance...
        name: %s
        ami id: %s
        instance type: %s
        key name: %s
        security group name: %s
        volume size: %s
        user data: %s""", args.name, args.ami_id, args.instance_type, args.key_name,
        args.security_group_name, args.volume_size, args.user_data)
    utils.run_instance(args.name, args.ami_id, args.instance_type, args.key_name,
                       args.security_group_name, args.volume_size, args.user_data)


def start(instance, args):
    utils.start_instance(instance)


def get(instance, args):
    logging.info(
        """Getting files from EC2 instance...
        instance id: %s
        source: %s
        target: %s""", args.instance_id, args.source, args.target)
    utils.get(instance, args.source, args.target)


def put(instance, args):
    logging.info(
        """Putting files on EC2 instance...
        instance id: %s
        source: %s
        target: %s
        exclude: %s""", args.instance_id, args.source, args.target, args.exclude)
    utils.put(instance, args.source, args.target, exclude=tuple(args.exclude))


def execute(instance, args):
    logging.info(
        """Executing script on EC2 instance...
        instance id: %s
        script: %s
        script arguments: %s""", args.instance_id, args.script, args.script_args)
    utils.exec_script(instance, args.script, args.script_args)


def stop(instance, args):
    utils.stop_instance(instance)


def info(instance, args):
    return utils.get_info(instance, args.field)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "action",
        type=str,
        choices=["create-image", "start", "launch", "info", "get", "put", "exec", "stop"],
        help="",
    )
    argparser.add_argument("--instance-id", type=str, help="", default="")
    # create image arguments
    argparser.add_argument("--name", type=str, default="CARLA_RLLIB")
    #argparser.add_argument("--base-ami-id", type=str, help="", default="ami-0dd9f0e7df0f0a138")
    argparser.add_argument("--base-ami-id", type=str, help="", default="")
    argparser.add_argument("--key-name", type=str, default=None)
    argparser.add_argument("--security-group-name", type=str, default=None)
    argparser.add_argument("--installation-scripts", nargs="+", type=str, default=[])
    # launch arguments
    argparser.add_argument("--ami-id", type=str,  default="ami-07006dbdc25cdc232")
    argparser.add_argument("--instance-type", type=str, default="t2.micro")
    argparser.add_argument("--volume-size", type=int, default=150)
    argparser.add_argument("--user-data", type=str, default="")
    # info arguments
    argparser.add_argument("--field", type=str, default="")
    # put&get arguments
    argparser.add_argument("--source", type=str, default=".")
    argparser.add_argument("--target", type=str, default=".")
    argparser.add_argument("--exclude", nargs='+', type=str, default=[])
    # exec arguments
    argparser.add_argument("--script", type=str, default=None)
    argparser.add_argument("--script-args", type=str, default="")

    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    arguments = argparser.parse_args()

    if arguments.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    result = ""
    if arguments.action == "create-image" or arguments.action == "launch":
        if arguments.name is None:
            raise RuntimeError("mandatory name attribute missing")
        arguments.key_name = arguments.name if arguments.key_name is None else arguments.key_name
        arguments.security_group_name = arguments.name if arguments.security_group_name is None else arguments.security_group_name

        if arguments.action == "create-image":
            if not arguments.base_ami_id:
                # arguments.base_ami_id = "ami-089d839e690b09b28"
                arguments.base_ami_id = DEFAULT_AMI[utils.get_region_name()]

            print(arguments.base_ami_id)
            result = create_ami(arguments)
        else:  # arguments.action == "launch"
            launch(arguments)
    else:
        if arguments.instance_id is None:
            raise RuntimeError("mandatory instance_id attribute missing")

        instance = utils.get_instance(arguments.instance_id)
        assert instance is not None

        if arguments.action == "start":
            result = start(instance, arguments)
        elif arguments.action == "info":
            result = info(instance, arguments)
        elif arguments.action == "get":
            result = get(instance, arguments)
        elif arguments.action == "put":
            result = put(instance, arguments)
        elif arguments.action == "exec":
            result = execute(instance, arguments)
        else:  # arguments.action == "stop":
            result = stop(instance, arguments)

    # result="hello"
    sys.exit(result)
