#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import hashlib
import math
import pygame
import time
from threading import Thread

import carla

from rllib_integration.sensors.sensor import PseudoSensor


# ==============================================================================
# -- Constants -----------------------------------------------------------------
# ==============================================================================

# Colors

# We will use the color palette used in Tango Desktop Project (Each color is indexed depending on brightness level)
# See: https://en.wikipedia.org/wiki/Tango_Desktop_Project

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)

COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)

COLOR_PURPLE = pygame.Color(186, 85, 211)

# ==============================================================================
# -- MapImage ------------------------------------------------------------------
# ==============================================================================

class MapImage(object):
    """
    Class encharged of rendering a 2D image from top view of a carla world (with pygame surfaces).
    A cache system is used, so if the OpenDrive content of a Carla town has not changed,
    it will read and use the stored image if it was rendered in a previous execution
    """

    def __init__(self, carla_world, carla_map, pixels_per_meter):
        """Renders the map image with all the information about the road network"""
        # TODO: The math.sqrt(2) is a patch due to the later rotation of this image
        self._pixels_per_meter = pixels_per_meter / math.sqrt(2)

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        # Cap the maximum zoom
        width_in_pixels = (1 << 14) - 1
        surface_pixel_per_meter = int(width_in_pixels / self.width)
        if surface_pixel_per_meter > self._pixels_per_meter:
            surface_pixel_per_meter = self._pixels_per_meter

        self._pixels_per_meter = surface_pixel_per_meter
        width_in_pixels = int(self._pixels_per_meter * self.width)

        # Load OpenDrive content
        opendrive_content = carla_map.to_opendrive()

        # Get hash based on content
        hash_func = hashlib.sha1()
        hash_func.update(opendrive_content.encode("UTF-8"))
        opendrive_hash = str(hash_func.hexdigest())

        # Build path for saving or loading the cached rendered map
        try:
            map_name = carla_map.name.split("/")[-1]
        except Exception:
            map_name = carla_map.name
        filename = map_name + "_" + opendrive_hash + ".tga"
        self.dirname = "map_cache"
        self.full_path = str(os.path.join(self.dirname, filename))

        if os.path.isfile(self.full_path):
            # Load image and scale it to the desired size
            self.big_map_surface = pygame.image.load(self.full_path)
            self.big_map_surface = pygame.transform.scale(self.big_map_surface, (width_in_pixels, width_in_pixels))

        else:
            # Render map
            self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels))
            self.draw_road_map(self.big_map_surface, carla_world, carla_map, precision=0.05)

            # To avoid race conditions between multiple ray workers.
            try:
                os.makedirs(self.dirname)
            except FileExistsError:
                pass

            # Remove files if selected town had a previous version saved
            list_filenames = glob.glob(os.path.join(self.dirname, carla_map.name) + "*")
            for town_filename in list_filenames:
                os.remove(town_filename)

            # Save rendered map for next executions of same map
            pygame.image.save(self.big_map_surface, self.full_path)

        # self.draw_road_map(self.big_map_surface, carla_world, carla_map, precision=0.05)
        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, precision=0.05):
        """Draws all the roads, including lane markings, arrows and traffic signs"""
        map_surface.fill(COLOR_ALUMINIUM_4)

        def lane_marking_color_to_tango(lane_marking_color):
            """Maps the lane marking color enum specified in PythonAPI to a Tango Color"""
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            """Draws solid lines in a surface given a set of points, width and color"""
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            """Draws broken lines in a surface given a set of points, width and color"""
            # Select which lines are going to be rendered from the set of lines
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]

            # Draw selected lines
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            """For multiple lane marking types (SolidSolid, BrokenSolid, SolidBroken and BrokenBroken), it converts them
             as a combination of Broken and Solid lines"""
            margin = 0.25
            marking_1 = [self.world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                return [(lane_marking_type, lane_marking_color, marking_1)]
            else:
                marking_2 = [self.world_to_pixel(lateral_shift(w.transform,
                                                          sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
                if lane_marking_type == carla.LaneMarkingType.SolidBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                    return [(carla.LaneMarkingType.Broken, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
                elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                    return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                            (carla.LaneMarkingType.Solid, lane_marking_color, marking_2)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane(surface, lane, color):
            """Renders a single lane in a surface and with a specified color"""
            for side in lane:
                lane_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in side]
                lane_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in side]

                polygon = lane_left_side + [x for x in reversed(lane_right_side)]
                polygon = [self.world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(surface, color, polygon, 5)
                    pygame.draw.polygon(surface, color, polygon)

        def draw_lane_marking(surface, waypoints):
            """Draws the left and right side of lane markings"""
            draw_lane_marking_single_side(surface, waypoints[0], -1)  # Left Side
            draw_lane_marking_single_side(surface, waypoints[1], 1)  # Right Side

        def draw_lane_marking_single_side(surface, waypoints, sign):
            """Draws the lane marking given a set of waypoints and decides whether drawing the right or left side of
            the waypoint based on the sign parameter"""
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            markings_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    # Get the list of lane markings to draw
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type

                    # Append each lane marking in the list
                    for marking in markings:
                        markings_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Add last marking
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                markings_list.append(marking)

            # Once the lane markings have been simplified to Solid or Broken lines, we draw them
            for markings in markings_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            """ Draws an arrow with a specified color given a transform"""
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir

            # Draw lines
            pygame.draw.lines(surface, color, False, [self.world_to_pixel(x) for x in [start, end]], 4)
            pygame.draw.lines(surface, color, False, [self.world_to_pixel(x) for x in [left, start, right]], 4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            """Draw stop traffic signs and its bounding box if enabled"""
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = self.world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [self.world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

        def lateral_shift(transform, shift):
            """Makes a lateral shift of the forward vector of a transform"""
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            """ Draws traffic signs and the roads network with sidewalks, parking and shoulders by generating waypoints"""
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            set_waypoints = []
            for waypoint in topology:
                waypoints = [waypoint]

                # Generate waypoints of a road id. Stop when road id differs
                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break
                set_waypoints.append(waypoints)

                # Draw Shoulders, Parkings and Sidewalks
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_4_5
                SIDEWALK_COLOR = COLOR_ALUMINIUM_3

                shoulder = [[], []]
                parking = [[], []]
                sidewalk = [[], []]

                for w in waypoints:
                    # Classify lane types until there are no waypoints by going left
                    l = w.get_left_lane()
                    while l and l.lane_type != carla.LaneType.Driving and not l.is_junction:

                        if l.lane_type == carla.LaneType.Shoulder:
                            shoulder[0].append(l)

                        if l.lane_type == carla.LaneType.Parking:
                            parking[0].append(l)

                        if l.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[0].append(l)

                        l = l.get_left_lane()

                    # Classify lane types until there are no waypoints by going right
                    r = w.get_right_lane()
                    while r and r.lane_type != carla.LaneType.Driving and not r.is_junction:

                        if r.lane_type == carla.LaneType.Shoulder:
                            shoulder[1].append(r)

                        if r.lane_type == carla.LaneType.Parking:
                            parking[1].append(r)

                        if r.lane_type == carla.LaneType.Sidewalk:
                            sidewalk[1].append(r)

                        r = r.get_right_lane()

                # Draw classified lane types
                draw_lane(map_surface, shoulder, SHOULDER_COLOR)
                draw_lane(map_surface, parking, PARKING_COLOR)
                draw_lane(map_surface, sidewalk, SIDEWALK_COLOR)

            # Draw Roads
            for waypoints in set_waypoints:
                waypoint = waypoints[0]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [self.world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Lane Markings and Arrows
                if not waypoint.is_junction:
                    draw_lane_marking(map_surface, [waypoints, waypoints])
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)

        actors = carla_world.get_actors()

        # Find and Draw Traffic Signs: Stops and Yields
        font_size = self.world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0), other_scale=1):
        """Converts the world coordinates to pixel coordinates"""
        x = self._pixels_per_meter * (location.x - self._world_offset[0]) * other_scale
        y = self._pixels_per_meter * (location.y - self._world_offset[1]) * other_scale
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        """Converts the world units to pixel units"""
        return int(self._pixels_per_meter * width)


# ==============================================================================
# -- BirdviewSensor ---------------------------------------------------------------------
# ==============================================================================

class BirdviewSensor(object):
    """Class that contains all the information of the carla world (in the form of pygame surfaces)"""

    def __init__(self, world, size, radius, hero):
        pygame.init()

        self.world = world
        self.town_map = self.world.get_map()
        self.radius = radius

        self.hero = hero
        self.hero_transform = self.hero.get_transform()
        self.pixels_per_meter = size / (2* self.radius)
        self.map_image = MapImage(self.world, self.town_map, self.pixels_per_meter)

        # Create the 'info' surfaces
        self.map_surface = self.map_image.surface  # Static elements
        map_surface_size = self.map_surface.get_width()
        self.actors_surface = pygame.Surface((map_surface_size, map_surface_size))  # Scene actors
        self.actors_surface.set_colorkey(COLOR_BLACK)  # Treat COLOR_BLACK pixels as transparent
        self.result_surface = pygame.Surface((map_surface_size, map_surface_size))  # Union of the previous two
        self.result_surface.set_colorkey(COLOR_BLACK)  # Treat COLOR_BLACK pixels as transparent

        # Create the 'egocentric' surface
        self.hero_surface = pygame.Surface((size, size))  # translation
        self.final_surface = pygame.Surface((size, size))  # rotation

    def _split_actors(self):
        """Splits the retrieved actors by type id"""
        vehicles = []
        traffic_lights = []
        speed_limits = []
        walkers = []

        for actor in self.world.get_actors():
            if 'vehicle' in actor.type_id:
                vehicles.append(actor)
            elif 'walker.pedestrian' in actor.type_id:
                walkers.append((actor))
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor)

        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_traffic_lights(self, surface, traffic_lights):
        """Renders the traffic lights and shows its triggers and bounding boxes if flags are enabled"""

        for tl in traffic_lights:
            pos = self.map_image.world_to_pixel(tl.get_location())
            radius = self.map_image.world_to_pixel_width(1.4)

            if tl.state == carla.TrafficLightState.Red:
                color = COLOR_SCARLET_RED_0
            elif tl.state == carla.TrafficLightState.Yellow:
                color = COLOR_BUTTER_0
            elif tl.state == carla.TrafficLightState.Green:
                color = COLOR_CHAMELEON_0
            elif tl.state == carla.TrafficLightState.Off:
                color = COLOR_ALUMINIUM_4
            else:
                color = COLOR_BLACK

            pygame.draw.circle(surface, color, (pos[0], pos[1]), radius)
            pygame.draw.circle(surface, COLOR_WHITE, (pos[0], pos[1]), radius, 1)

    def _render_speed_limits(self, surface, speed_limits, angle):
        """Renders the speed limits by drawing two concentric circles (outer is red and inner white) and a speed limit text"""

        font_size = self.map_image.world_to_pixel_width(2)
        radius = self.map_image.world_to_pixel_width(2)
        font = pygame.font.SysFont('Arial', font_size)

        for sl in speed_limits:

            x, y = self.map_image.world_to_pixel(sl.get_location())

            # Render speed limit concentric circles
            white_circle_radius = int(radius * 0.75)

            pygame.draw.circle(surface, COLOR_SCARLET_RED_1, (x, y), radius)
            pygame.draw.circle(surface, COLOR_ALUMINIUM_0, (x, y), white_circle_radius)

            limit = sl.type_id.split('.')[2]
            font_surface = font.render(limit, True, COLOR_ALUMINIUM_5)
            font_surface = pygame.transform.rotate(font_surface, -angle)
            offset = font_surface.get_rect(center=(x, y))
            surface.blit(font_surface, offset)

    def _render_walkers(self, surface, list_w):
        """Renders the walkers' bounding boxes"""
        for w in list_w:
            color = COLOR_PLUM_0

            # Compute bounding box points
            bb = w.bounding_box.extent
            corners = [
                2*carla.Location(x=-bb.x, y=-bb.y),
                2*carla.Location(x=bb.x, y=-bb.y),
                2*carla.Location(x=bb.x, y=bb.y),
                2*carla.Location(x=-bb.x, y=bb.y)]

            w.get_transform().transform(corners)
            corners = [self.map_image.world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v):
        """Renders the vehicles' bounding boxes"""
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v.attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v.attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v.bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y),
                       carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0),
                       carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y),
                       carla.Location(x=-bb.x, y=-bb.y)
                       ]
            v.get_transform().transform(corners)
            corners = [self.map_image.world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def render_actors(self, surface, angle):
        """Renders all the actors"""
        # Split the actors by vehicle type id
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()

        # Static actors
        self._render_traffic_lights(surface, traffic_lights)
        self._render_speed_limits(surface, speed_limits, angle)

        # Dynamic actors
        self._render_vehicles(surface, vehicles)
        self._render_walkers(surface, walkers)

    def get_data(self):
        """Renders the map and all the actors in hero and map mode"""
        self.hero_transform = self.hero.get_transform()

        # Together with pygame.Surface.set_colorkey, makes their background transparent
        self.result_surface.fill(COLOR_BLACK)
        self.actors_surface.fill(COLOR_BLACK)

        # Angle on with to rotate to make the view egocentric
        angle = self.hero_transform.rotation.yaw + 90.0

        # Render the actors
        self.render_actors(self.actors_surface, angle)

        # Get a point in front of the ego vehicle. It will act as the center of the resulting image.
        # Then clip the surfaces to render only the visible parts, improving perfomance.
        hero_center_location = self.hero_transform.location + self.hero_transform.get_forward_vector()*self.radius / 2
        hero_screen_location = self.map_image.world_to_pixel(hero_center_location)
        hero_surface_size = (self.hero_surface.get_width(), self.hero_surface.get_height())
        offset = (hero_screen_location[0] - hero_surface_size[0] / 2,
                 (hero_screen_location[1] - hero_surface_size[1] / 2))
        clipping_rect = pygame.Rect(offset[0], offset[1], hero_surface_size[0] * 2, hero_surface_size[1] * 2)

        self.map_surface.set_clip(clipping_rect)
        self.actors_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

        # Join map and actor surface
        self.result_surface.blit(self.map_surface, (0, 0))
        self.result_surface.blit(self.actors_surface, (0, 0))

        # Translate it to make the surface egocentric
        self.hero_surface.blit(self.result_surface, (-offset[0], -offset[1]))

        # Rotate it to make the image egocentric. Zoom by sqrt(2) to avoid seeing black corners
        rotated_surface = pygame.transform.rotozoom(self.hero_surface, angle, math.sqrt(2))
        center = (hero_surface_size[0] / 2, hero_surface_size[1] / 2)
        rotation_pivot = rotated_surface.get_rect(center=center)
        self.final_surface.blit(rotated_surface, rotation_pivot)

        # Get the surface into a numpy array
        array3d = pygame.surfarray.array3d(self.final_surface)
        array3d = array3d.swapaxes(0, 1)

        return array3d

    def destroy(self):
        """Destroy the hero actor when class instance is destroyed"""
        pygame.quit()


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper

class BirdviewManager(PseudoSensor):
    """
    This class is responsible of creating a 'birdview' pseudo-sensor, which is a simplified
    version of CARLA's non rendering mode.
    """

    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

        self.world = parent.get_world()
        self.running = False  # Flag to stop the execution of the sensor
        self.previous_frame = None

        # Get the sensor instance and run it
        self.sensor = BirdviewSensor(self.world, attributes["size"], attributes["radius"] , parent)
        self.run()

    @threaded
    def run(self):
        """Function to copy the functionality of CARLA sensor.listen() callback,
        responsible of sending the data of the sensor each tick"""
        self.running = True
        self.previous_frame = self.world.get_snapshot().frame
        while self.running:
            frame = self.world.get_snapshot().frame

            # Avoid getting the data more than once per frame
            if frame > self.previous_frame:
                self.callback(self.sensor.get_data(), frame)
                self.previous_frame = frame
            else:
                time.sleep(0.005)

    def destroy(self):
        """Stop the sensor and its execution"""
        self.running = False
        self.sensor.destroy()

    def parse(self, data):
        """Parses the data into the corresponfing format"""
        return data
