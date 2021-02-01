#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

def print_spawn_point(world):
    spawn_points = list(world.get_map().get_spawn_points())
    for i in range(len(spawn_points)):
        print(
            "spawn_point:",
            i,
            "x:",
            spawn_points[i].location.x,
            "y:",
            spawn_points[i].location.y,
            "z:",
            spawn_points[i].location.z,
            "pitch:",
            spawn_points[i].rotation.pitch,
            "roll:",
            spawn_points[i].rotation.roll,
            "yaw:",
            spawn_points[i].rotation.yaw,
        )


def draw_spawn_points(world):
    spawn_points = list(world.get_map().get_spawn_points())
    i = 0
    for spawn_point in spawn_points:
        world.debug.draw_point(
            spawn_point.location, size=0.1, life_time=1000.0, persistent_lines=True
        )
        name = (
            str(i)
            + "   ,   "
            + str(round(spawn_point.location.x, 1))
            + "   ,   "
            + str(round(spawn_point.location.y, 1))
            + "   ,   "
            + str(round(spawn_point.location.z, 1))
        )
        world.debug.draw_string(
            spawn_point.location, name, life_time=100.0, persistent_lines=True
        )
        i = i + 1


def get_actor_status(actor):
    position = actor.get_transform()
    velocity = actor.get_velocity()
    control = actor.get_control()
    heading = "N" if abs(position.rotation.yaw) < 89.5 else ""
    heading += "S" if abs(position.rotation.yaw) > 90.5 else ""
    heading += "E" if 179.5 > position.rotation.yaw > 0.5 else ""
    heading += "W" if -0.5 > position.rotation.yaw > -179.5 else ""
    return position, velocity, control, heading


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + u"\u2026") if len(name) > truncate else name


def print_blueprint_attributes(blueprint_library):
    for blueprint in blueprint_library:
        print(blueprint)
        for attribute in blueprint:
            print("  - %s" % attribute)


def split_actors(actors):
    vehicles = []
    traffic_lights = []
    speed_limits = []
    walkers = []

    for actor in actors:
        if "vehicle" in actor.type_id:
            vehicles.append(actor)
        elif "traffic_light" in actor.type_id:
            traffic_lights.append(actor)
        elif "speed_limit" in actor.type_id:
            speed_limits.append(actor)
        elif "walker" in actor.type_id:
            walkers.append(actor)

    return vehicles, traffic_lights, speed_limits, walkers
