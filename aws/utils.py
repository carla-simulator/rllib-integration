#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
boto3 ec2 wrappers.
"""

import datetime
import functools
import logging
import os
import shutil
import stat
import sys
import time

import boto3
from botocore.exceptions import ClientError

import paramiko
import scp

MAX_TRIES = 5

def create_key_pair(key_name, private_key_file_name=None):
    """
    Creates a key pair that can be used to securely connect to an Amazon EC2 instance.

    :param key_name: The name of the key pair to create.
    :type key_name: string
    :param private_key_file_name: The file name where the private key portion is stored.
    :type private_key_file_name: string
    :return: The newly created key pair.
    :rtype: ec2.KeyPair
    """
    ec2 = _get_resource("ec2")
    try:
        key_pair = ec2.create_key_pair(KeyName=key_name)
        logging.info("Created key %s.", key_name)
        if private_key_file_name is None:
            private_key_file_name = os.path.join(_get_folder_keys(), key_name + ".pem")
        with open(private_key_file_name, 'w') as pk_file:
            pk_file.write(key_pair.key_material)
        os.chmod(private_key_file_name, stat.S_IREAD)
        logging.info("Wrote private key to %s.", private_key_file_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidKeyPair.Duplicate":
            logging.info(e.response["Error"]["Message"])
            return
        else:
            logging.exception("Couldn't create key %s.", key_name)
            raise
    else:
        return key_pair


def create_security_group(group_name, group_description=None):
    """
    Creates a security group in the default virtual private cloud (VPC) of the current account,
    then adds rules to the security group to allow access to HTTP, HTTPS, SSH.  

    :param group_name: The name of the security group to create.
    :type group_name: string
    :param group_description: The description of the security group to create.
    :param group_description: string
    :return: The newly created security group.
    :rtype: ec2.SecurityGroup
    """
    ec2 = _get_resource("ec2")
    try:
        default_vpc = list(ec2.vpcs.filter(Filters=[{'Name': 'isDefault', 'Values': ['true']}]))[0]
        logging.info("Got default VPC %s.", default_vpc.id)
    except ClientError as e:
        logging.exception("Couldn't get VPCs.")
        raise
    except IndexError:
        logging.exception("No default VPC in the list.")
        raise

    try:
        if group_description is None:
            group_description = "generated on {date:%Y-%m-%d %H:%M:%S} by {script:}".format(
                date=datetime.datetime.now(), script=os.path.basename(__file__))
        security_group = default_vpc.create_security_group(GroupName=group_name,
                                                           Description=group_description)
        logging.info("Created security group %s in VPC %s.", group_name, default_vpc.id)
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidGroup.Duplicate":
            logging.info(e.response["Error"]["Message"])
            return
        else:
            logging.exception("Couldn't create security group %s.", group_name)
            raise

    try:
        security_group.authorize_ingress(IpPermissions=[
            {
                # Ray dashboard
                'IpProtocol': 'tcp',
                'FromPort': 8265,
                'ToPort': 8300,
                'IpRanges': [{
                    'CidrIp': '0.0.0.0/0'
                }]
            },
            {
                # Ray dashboard
                'IpProtocol': 'tcp',
                'FromPort': 6006,
                'ToPort': 6006,
                'IpRanges': [{
                    'CidrIp': '0.0.0.0/0'
                }]
            },
            {
                # SSH ingress open to anyone
                'IpProtocol': 'tcp',
                'FromPort': 22,
                'ToPort': 22,
                'IpRanges': [{
                    'CidrIp': '0.0.0.0/0'
                }]
            }
        ])
        logging.info("Set inbound rules for %s", security_group.id)
    except ClientError:
        logging.exception("Couldn't authorize inbound rules for %s.", group_name)
        raise
    else:
        return security_group


def get_instance(instance_id):
    ec2 = _get_resource("ec2")
    instances = list(ec2.instances.filter(InstanceIds=[instance_id]))
    if not instances:
        return None
    assert len(instances) == 1
    return instances[0]


def print_image_info(image):
    print("\033[1mImage id:\033[0m {}".format(image.id))


def print_instance_info(instance):
    print("\033[1mInstance id:\033[0m {}".format(instance.id))
    print("\033[1mKey:\033[0m {}".format(instance.key_name))
    print("\033[1mSecurity group:\033[0m ")
    for group in instance.security_groups:
        group_name, group_id = group["GroupName"], group["GroupId"]
        print("\t{} ({})".format(group_name, group_id))


def get_info(instance, field):
    if field == "public_ip":
        return instance.public_ip_address
    elif field == "pem_file":
        return _get_key_filename(instance)
    else:
        return None

def run_instance(name, image_id, instance_type, key_name, security_group_name, volume_size=10, user_data=""):
    """
    Creates a new Amazon EC2 instance. The instance automatically starts immediately after it is
    created. The instance is created in the default VPC of the current account.
    
    :param name: The name of the EC2 instance.
    :type name: string
    :param image_id: The Amazon Machine Image (AMI) id.
    :type image_id: string
    :param instance_type: The type of instance to create, such as 't2.micro'.
    :type instance_type: string
    :param key_name: The name of the key pair that is used to secure connections to the instance.
    :type key_name: string
    :param security_group_name: The security group name to grant access to the instance.
    :type security_group_name: string
    :param volume_size: Root EBS volume size
    :type volume_size: int
    :param user_data: The user data to make available to the instance.
    :type user_data: string
    :return: The newly created instance.
    :rtype: ec2.Instance
    """
    ec2 = _get_resource("ec2")
    try:
        create_key_pair(key_name)
        create_security_group(security_group_name)

        logging.info(
            "Running EC2 instance with image id %s and instance type % s. This may take a while...",
            image_id, instance_type)
        instance = ec2.create_instances(ImageId=image_id,
                                        InstanceType=instance_type,
                                        KeyName=key_name,
                                        SecurityGroups=[security_group_name],
                                        BlockDeviceMappings=[
                                            {
                                                'DeviceName': '/dev/sda1',
                                                'Ebs': {
                                                    'VolumeSize': volume_size,
                                                },
                                            },
                                        ],
                                        Placement={
                                            'AvailabilityZone': 'eu-west-3c',
                                        },
                                        UserData=user_data,
                                        MinCount=1,
                                        MaxCount=1)[0]
        ec2.create_tags(Resources=[instance.id], Tags=[{"Key": "Name", "Value": name}])

        instance.wait_until_running()
        # ensure we can connect via ssh
        time.sleep(30)

        instance = list(ec2.instances.filter(InstanceIds=[instance.id]))[0]
        logging.info("Created instance %s. Public IPv4 address: %s", instance.id,
                     instance.public_ip_address)
    except ClientError:
        logging.exception("Couldn't create instance with image %s, instance type %s, and key %s.",
                          image_id, instance_type, key_name)
        raise
    else:
        return instance


def start_instance(instance):
    logging.info("Starting EC2 instance %s", instance.id)
    instance.start()
    instance.wait_until_running()
    # cleaning cache because the public ip address of the instance has been modified.
    _clear_cache()


def stop_instance(instance):
    logging.info("Stopping EC2 instance %s", instance.id)
    instance.stop()
    instance.wait_until_stopped()


def get_image(name):
    ec2 = _get_resource("ec2")
    images = list(ec2.images.filter(Filters=[{'Name': 'name', 'Values': [name]}], Owners=["self"]))
    if images:
        assert len(images) == 1
        return images[0]
    return None


def create_image(name,
                 base_image_id,
                 instance_type,
                 key_name,
                 security_group_name,
                 volume_size=10,
                 installation_scripts=(),
                 user_data=""):
    ec2 = _get_resource("ec2")
    instance = run_instance(name, base_image_id, instance_type, key_name, security_group_name, volume_size, user_data)

    for script in installation_scripts:
        logging.info("Executing installation script %s", os.path.basename(script))
        exec_script(instance, script)

    logging.info("Creating image from EC2 instance %s. This may take a while...", instance.id)
    image = instance.create_image(Name=name)
    while image.state == "pending":
       time.sleep(5)
       image = list(ec2.images.filter(ImageIds=[image.id]))[0]
    if image.state == "available":
       logging.info("Image created with id %s", image.id)
    else:
       logging.error("Couldn't create image from EC2 instance %s", instance.id)

    stop_instance(instance)

    return instance, image


def exec_script(instance, script, args="", rsync_folder=False):
    logging.info("Executing script %s in EC2 instance %s", os.path.basename(script), instance.id)
    command = ""
    if not os.path.isfile(script):
        logging.error("The provided script '%s' is not a file.", script)
        return
    if rsync_folder:
        put(instance, os.path.dirname(script))
        folder = os.path.basename(os.path.dirname(script))
        command += " cd {} && ".format(folder)
    else:
        put(instance, script)
    command += "./{} {}".format(os.path.basename(script), args)
    logging.info("Running: {}".format(command))
    exec_command(instance, command)


def exec_command(instance, command):
    ssh_client = _get_ssh_client(instance)
    _, stdout, stderr = ssh_client.exec_command(command)
    for line in iter(stdout.readline, ""):
        print(line, end="")

def put(instance, source, target=".", exclude=(".git", "keys", "__pycache__", "map_cache")):
    logging.info("Copying %s into EC2 instance %s", source, instance.id)
    scp_client = _get_scp_client(instance)
    if os.path.isdir(source):
        folder = os.path.basename(source)
        target = os.path.join(target, folder)
        exec_command(instance, "mkdir -p {}".format(target))

        base, dirs, files = next(os.walk(source))
        dirs = [os.path.join(base, dir_) for dir_ in dirs if dir_ not in exclude]
        files = [os.path.join(base, file) for file in files]
        for dir_ in dirs:
            remote_path = os.path.join(target, os.path.basename(dir_))
            scp_client.put(dir_, remote_path=remote_path, recursive=True)
        scp_client.put(files, remote_path=target)
    else:
        scp_client.put(files=source, remote_path=target)


def get(instance, remote_path, local_path=".", recursive=True):
    logging.info("Getting %s from EC2 instance %s", remote_path, instance.id)
    scp_client = _get_scp_client(instance)
    scp_client.get(remote_path=remote_path, local_path=local_path, recursive=recursive)


def _get_username(instance):
    return "ubuntu"


def _get_folder_keys():
    return os.path.join(os.path.expanduser("~"), "rllib_keys") 


def _get_key_filename(instance):
    folder = _get_folder_keys()
    pem_file = os.path.join(folder, instance.key_name + ".pem")
    if not os.path.isfile(pem_file):
        user_pem_file = input("pem file for key {} not found. Please, provide the appropriate file: ".format(instance.key_name))
        if not os.path.exists(folder):
            os.mkdir(folder) 
        shutil.copyfile(user_pem_file, pem_file)
        os.chmod(pem_file, stat.S_IREAD)
    return pem_file


def _clear_cache():
    _get_ssh_client.cache_clear()
    _get_scp_client.cache_clear()


@functools.lru_cache()
def _get_ssh_client(instance):
    hostname = instance.public_ip_address
    username = _get_username(instance)
    key_filename = _get_key_filename(instance)
    logging.info("Connecting to instance %s. Public IPv4 address: %s", instance.id,
                 instance.public_ip_address)

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    i, is_connected = 0, False
    while not is_connected and i <= MAX_TRIES:
        try:
            ssh_client.connect(hostname=hostname, username=username, key_filename=key_filename)
            is_connected = True
        except:
            logging.warning("Unable to connect to the instance. Trying again...")
            time.sleep(5)
            i+= 1

    return ssh_client


@functools.lru_cache()
def _get_scp_client(instance):
    def progress(filename, size, sent):
        sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))

    ssh_client = _get_ssh_client(instance)
    return scp.SCPClient(ssh_client.get_transport(), progress=progress)


@functools.lru_cache()
def _get_resource(name):
    return boto3.resource(name)


@functools.lru_cache()
def get_region_name():
    session = boto3.session.Session()
    return session.region_name
