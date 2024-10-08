#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pygame as pg
import pygame.transform
from pygame.sprite import Sprite
from pygame.transform import scale
from pygame import Surface
import numpy as np
from math import copysign
import colormap


ANIMATION_SPEED = 2  # 1 is normal speed
ALPHA_INACTIVE = 0.5
ALPHA_ACTIVE = 1.0
COLORS = {
    'text': (255, 255, 255),
    'bg': (255, 255, 255),
    'connection': (54, 86, 181)
}
DUMP_SPEED = 1e-2
DUMP_FORCE = 1e-3
EPS = 1e-24
# left, right, up, down
ACTIONS = ['l', 'r', 'u', 'd']


def convert_color(color):
    return tuple([int(round(255*x)) for x in color])


class State(Sprite):
    def __init__(
            self,
            info: tuple[str, float, float, float, tuple[int, int, int]],
            image: Surface,
            position: tuple[int, int],
            bsize: int,
            *groups,
            additional_info: list = None,
            custom_color: tuple[int, int, int] = None,
            custom_color_mask: Surface = None
    ):
        """
            info: one float number is for relative size
             and the other for the shift relative to the center
        """
        super().__init__(*groups)
        # rescale to fit size
        original_size = image.get_rect().size
        ratio = original_size[0] / original_size[1]
        if ratio < 1:
            size = (int(bsize * ratio), bsize)
        else:
            size = (bsize, int(bsize/ratio))

        if custom_color is not None:
            if custom_color_mask is not None:
                bg_image = Surface(original_size).convert_alpha()
                bg_image.fill(custom_color)
                bg_image.blit(custom_color_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                image.blit(bg_image, (0, 0))

        self.image = scale(image, size)
        self.rect = self.image.get_rect()
        self.rect.center = position

        self.decay = (1 / 100) * ANIMATION_SPEED

        text_size = int(bsize * info[1])
        text_vpos = int(size[1]//2 + bsize * info[2])
        text_hpos = int(size[0]//2 + bsize * info[3])

        text = pg.font.SysFont(
            'arial', text_size
        ).render(info[0], True, info[-1])
        text_rect = text.get_rect()
        text_rect.center = (text_hpos, text_vpos)

        self.image.blit(text, text_rect)
        if additional_info is not None:
            for txt in additional_info:
                text_size = int(bsize * txt[1])
                text_vpos = int(size[1] // 2 + bsize * txt[2])
                text_hpos = int(size[0] // 2 + bsize * txt[3])
                text = pg.font.SysFont(
                    'arial', text_size
                ).render(txt[0], True, txt[-1])
                text_rect = text.get_rect()
                text_rect.center = (text_hpos, text_vpos)
                self.image.blit(text, text_rect)

        self.active = False
        self.destroyed = False
        self.alpha = 0
        self.target_alpha = ALPHA_INACTIVE
        self.image.set_alpha(int(round(255 * self.alpha)))

        self.to_scroll = 0
        self.scroll_frames = int(round(60 / ANIMATION_SPEED))
        self.scroll_dx = 0

    def update(self, *args, **kwargs):
        # scrolling
        if self.to_scroll > 0:
            if self.to_scroll < abs(self.scroll_dx):
                scroll_dx = copysign(self.to_scroll, self.scroll_dx)
                self.to_scroll = 0
            else:
                scroll_dx = self.scroll_dx
                self.to_scroll -= abs(self.scroll_dx)

            self.rect.move_ip(-scroll_dx, 0)

        # activate/deactivate animation
        self.update_alpha()

        if self.destroyed and (self.alpha == 0):
            self.kill()

    def update_alpha(self):
        delta = self.target_alpha - self.alpha
        self.alpha += np.sign(delta) * min(abs(delta), self.decay)
        self.alpha = np.clip(self.alpha, 0.0, 1.0)
        self.image.set_alpha(int(round(255 * self.alpha)))

    def activate(self):
        self.target_alpha = ALPHA_ACTIVE
        self.active = True

    def deactivate(self):
        self.target_alpha = ALPHA_INACTIVE
        self.active = False

    def destroy(self):
        self.target_alpha = 0
        self.destroyed = True


class LearningHistory:
    def __init__(
            self,
            canvas_size: tuple[int, int],
            step_size: tuple[int, int],
            sprites: tuple,
            masks: tuple = None,
            custom_cmap=None,
            debug=True
    ):
        self.sprites = sprites
        self.masks = masks
        self.custom_cmap = custom_cmap
        self.horizontal_step, self.vertical_step = step_size
        self.canvas = Surface(canvas_size)
        self.canvas.fill(COLORS['bg'])
        self.bgd = Surface(canvas_size)
        self.bgd.fill(COLORS['bg'])
        self.debug = debug
        self.debug_text = pg.font.SysFont(
                    'notosansmono', 12
                )

        self.center = (canvas_size[0]//2, canvas_size[1]//2)

        self.current_step = 0
        self.current_step_view = 0
        self.step_groups = list()
        self.main_group = pg.sprite.Group()

        self.scroll_frames = int(round(60 / ANIMATION_SPEED))

    def handle(self, event):
        if self.debug:
            print(event)
            d_text = self.debug_text.render(
                str(event), True, (0, 0, 0)
            )
            bg = Surface((self.canvas.get_size()[0], d_text.get_size()[1]))
            bg.fill(COLORS['bg'])
            self.canvas.blit(bg, (0, 0))
            self.canvas.blit(d_text, (0, 0))

        event_type = event[0]
        if event_type == 'new_obs':
            pos, obs_state, action = event[1:]

            self.set_position(pos)

            self.step_groups.append(
                {
                    'predicted': [],
                    'clones': []
                }
            )

            if (self.custom_cmap is not None) and (self.masks is not None):
                custom_color = convert_color(self.custom_cmap(obs_state))
                color_mask = self.masks[0]
            else:
                custom_color = None
                color_mask = None

            state_obj = State(
                (str(obs_state), 0.4, -0.1, 0, COLORS['text']),
                self.sprites[0],
                self.center,
                min(self.horizontal_step, self.vertical_step),
                self.main_group,
                additional_info=[(ACTIONS[action], 0.2, 0.3, 0, COLORS['text'])],
                custom_color=custom_color,
                custom_color_mask=color_mask
            )
            state_obj.activate()
        elif event_type == 'set_state':
            self.set_position(self.current_step)
            state = event[1]
            self.set_state(state, self.current_step)
        elif event_type == 'move':
            pos = event[1]
            self.set_position(pos)
        elif event_type == 'set_prev_state':
            self.set_position(self.current_step)
            state = event[1]
            self.set_state(state, self.current_step - 1)
            self.clear_prediction(self.current_step)
        elif event_type == 'predict_forward':
            self.set_position(self.current_step)
            correct, wrong = event[1:]
            self.clear_prediction(self.current_step)

            for clone, obs_state, weight in correct:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step, True, False
                )

            for clone, obs_state, weight in wrong:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step, False, False
                )

        elif event_type == 'predict_backward':
            self.set_position(self.current_step)
            correct, wrong = event[1:]
            self.clear_prediction(self.current_step-1)

            for clone, obs_state, weight in correct:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step-1,
                    True, False, False
                )

            for clone, obs_state, weight in wrong:
                self.set_prediction(
                    clone, obs_state, weight, self.current_step-1,
                    False, False, False
                )
        elif event_type == 'reset':
            for sp in self.main_group:
                sp.destroy()
            self.step_groups.clear()
            self.current_step = 0

    def clear_prediction(self, step):
        group = self.step_groups[step]['predicted']
        if len(group) > 0:
            for x in group:
                x.destroy()
            group.clear()

    def set_state(self, state, pos):
        group = self.step_groups[pos]['clones']
        if len(group) > 0:
            group[-1].deactivate()
        shift_y = int(0.8*self.vertical_step) + len(group) * int(0.5 * self.vertical_step)
        shift_x = (self.current_step - pos) * self.horizontal_step
        state_obj = State(
            (str(state), 0.5, 0, 0, COLORS['text']),
            self.sprites[1],
            (self.center[0] - shift_x, self.center[1] - shift_y),
            min(self.horizontal_step, int(0.5 * self.vertical_step)),
            self.main_group
        )
        group.append(state_obj)
        state_obj.activate()

    def set_prediction(self, clone, obs_state, weight, pos, correct, temp, forward=True):
        sprite = self.sprites[2 + int(temp)]
        h_shifts = (0.05, -0.4)
        if not forward:
            sprite = pygame.transform.flip(sprite, True, False)
            h_shifts = (-h_shifts[0], -h_shifts[1])
        group = self.step_groups[pos]['predicted']
        shift_y = int(0.8*self.vertical_step) + len(group) * int(0.5 * self.vertical_step)
        shift_x = (self.current_step - pos) * self.horizontal_step

        if (self.custom_cmap is not None) and (self.masks is not None):
            custom_color = convert_color(self.custom_cmap(obs_state))
            color_mask = self.masks[1]
            if not forward:
                color_mask = pygame.transform.flip(color_mask, True, False)
        else:
            custom_color = None
            color_mask = None

        state_obj = State(
            (str(clone), 0.4, -0.1, h_shifts[0], COLORS['text']),
            sprite,
            (self.center[0] - shift_x, self.center[1] + shift_y),
            min(self.horizontal_step, int(0.5 * self.vertical_step)),
            self.main_group,
            additional_info=[
                (str(obs_state), 0.2, 0.26, h_shifts[0], COLORS['text']),
                (str(weight), 0.2, -0.25, h_shifts[1], (0, 0, 0))
            ],
            custom_color=custom_color,
            custom_color_mask=color_mask
        )
        group.append(state_obj)
        if correct:
            state_obj.activate()

    def set_position(self, pos):
        to_scroll = abs(pos - self.current_step_view) * self.horizontal_step
        scroll_dx = (
                np.sign(pos - self.current_step_view) *
                max(1, int(round(to_scroll / self.scroll_frames)))
        )
        for sprite in self.main_group:
            sprite.to_scroll = to_scroll
            sprite.scroll_dx = scroll_dx
        self.current_step = pos
        self.current_step_view = pos

    def scroll(self, delta_pos):
        to_scroll = abs(delta_pos) * self.horizontal_step
        scroll_dx = (
                np.sign(delta_pos) * to_scroll
        )
        for sprite in self.main_group:
            sprite.to_scroll = to_scroll
            sprite.scroll_dx = scroll_dx
        self.current_step_view += delta_pos

    def update(self):
        self.main_group.clear(self.canvas, self.bgd)
        self.main_group.draw(self.canvas)
        self.main_group.update()


class Node(State):
    def __init__(
            self,
            clone: int,
            obs_state: int,
            position: tuple[int, int],
            image: Surface,
            bsize: int,
            dilation: float,
            speed_decay: float,
            *groups,
            custom_color=None,
            custom_color_mask=None
    ):
        info = (str(clone), 0.4, -0.1, 0, COLORS['text'])
        additional_info = [(str(obs_state), 0.2, 0.25, 0, COLORS['text'])]
        super().__init__(
            info, image, position, bsize, *groups, additional_info=additional_info,
            custom_color_mask=custom_color_mask, custom_color=custom_color
        )
        self.clone = clone
        self.obs_state = obs_state
        self.radius = bsize * dilation

        self.speed_decay = speed_decay
        self.velocity = [0.0, 0.0]
        self.position = list(position)

        self.activate()

    def move(self, vector):
        self.velocity[0] += vector[0]
        self.velocity[1] += vector[1]
        self.velocity[0] *= self.speed_decay
        self.velocity[1] *= self.speed_decay

        if abs(self.velocity[0]) <= DUMP_SPEED:
            self.velocity[0] = 0
        if abs(self.velocity[1]) <= DUMP_SPEED:
            self.velocity[1] = 0

    def update(self, *args, **kwargs):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        self.rect.center = (
            int(round(self.position[0])),
            int(round(self.position[1]))
        )
        super().update(*args, **kwargs)


class TransitionGraph:
    def __init__(
            self,
            canvas_size: tuple[int, int],
            sprites: tuple,
            node_size: int,
            force: float,
            force_decay: float = 0.9,
            speed_decay: float = 0.9,
            dilation: float = 1.5,
            rad_decay: float = 1.0,
            rep_factor: float = 1.0,
            att_factor: float = 1.0,
            gravitation: float = 0.1,
            masks: tuple = None,
            custom_cmap=None
    ):
        self.node_size = node_size
        self.dilation = dilation
        self.sprites = sprites
        self.masks = masks
        self.custom_cmap = custom_cmap
        self.canvas = Surface(canvas_size)
        self.canvas.fill(COLORS['bg'])
        self.bgd = Surface(canvas_size)
        self.bgd.fill(COLORS['bg'])
        self.connection_font = pg.font.SysFont('arial',  int(node_size * 0.2))
        self.connection_cmap = colormap.Colormap().cmap_bicolor(
            'white', 'blue'
        )

        self.center = (canvas_size[0]//2, canvas_size[1]//2)

        self.force = force
        self.init_force = force
        self.force_decay = force_decay
        self.speed_decay = speed_decay

        self.rep_factor = rep_factor
        self.att_factor = att_factor
        self.rad_factor = 1.0
        self.init_rad_factor = 1.0
        self.rad_decay = rad_decay
        self.gravitation = gravitation
        self.init_gravitation = gravitation

        self.safe_margin = 3
        self.last_pos = self.center

        self.current_step = 0
        self.main_group = pg.sprite.Group()

        self.vertices = dict()
        self.edges = dict()

    def handle(self, event):
        event_type = event[0]
        if event_type == 'reinforce_con':
            self.force = self.init_force
            self.gravitation = self.init_gravitation
            self.rad_factor = self.init_rad_factor

            prev_action, prev_state, state = event[1:]

            node1 = f'{prev_state}'
            node2 = f'{state}'
            edge = f'{prev_state}_{state}'

            if not (edge in self.edges):
                self.edges[edge] = {
                    'node1': node1,
                    'node2': node2,
                    'actions': {a: 0 for a in ACTIONS},
                    'strength': 1
                }
            self.edges[edge]['actions'][ACTIONS[prev_action]] += 1

            for node, s in zip((node1, node2), (prev_state, state)):
                if not (node in self.vertices):
                    # sample random position
                    position = (
                            np.asarray(self.last_pos) +
                            np.random.uniform(-self.node_size, self.node_size, size=2)
                    )
                    color = convert_color(self.custom_cmap(s[1]))
                    self.vertices[node] = {
                        'vis': Node(
                            s[0],
                            s[1],
                            (int(position[0]), int(position[1])),
                            self.sprites[0],
                            self.node_size,
                            self.dilation,
                            self.speed_decay,
                            self.main_group,
                            custom_color=color,
                            custom_color_mask=self.masks[2]
                        ),
                        'in_edges': set(),
                        'out_edges': set(),
                    }
                self.last_pos = self.vertices[node]['vis'].rect.center

            self.vertices[node1]['out_edges'].add(edge)
            self.vertices[node2]['in_edges'].add(edge)

            self.normalize_edges()

        elif event_type == 'remove_con':
            self.force = self.init_force
            self.gravitation = self.init_gravitation
            self.rad_factor = self.init_rad_factor

            prev_action, prev_state, state = event[1:]
            edge = f'{prev_state}_{state}'
            if edge in self.edges:
                actions = self.edges[edge]['actions']
                actions[ACTIONS[prev_action]] -= 1
                if sum(actions.values()) == 0:
                    self.vertices[self.edges[edge]['node1']]['out_edges'].remove(edge)
                    self.vertices[self.edges[edge]['node2']]['in_edges'].remove(edge)
                    self.edges.pop(edge)

            self.normalize_edges()
        elif event_type == 'punish_con':
            self.force = self.init_force
            self.gravitation = self.init_gravitation
            self.rad_factor = self.init_rad_factor

            prev_action, prev_state, state = event[1:]

            edge = f'{prev_state}_{state}'
            if edge in self.edges:
                self.edges[edge]['actions'][ACTIONS[prev_action]] -= 1
                self.normalize_edges()

    def update(self):
        self.canvas.blit(self.bgd, (0, 0))

        # draw edges
        for edge in self.edges.values():
            start_pos = self.vertices[edge['node1']]['vis'].rect.center
            end_pos = self.vertices[edge['node2']]['vis'].rect.center
            color = self.connection_cmap(int(round(255*edge['strength'])))
            color = tuple([int(round(x*255)) for x in color])
            label = self.connection_font.render(
                ' '.join([f'{x}:{v}' for x, v in edge['actions'].items() if v > 0]),
                True,
                color
            )

            if start_pos != end_pos:
                middle_pos = ((start_pos[0] + end_pos[0])//2, (start_pos[1] + end_pos[1])//2)
                direction = np.sign(end_pos[0] - start_pos[0])
                label_pos = (middle_pos[0], middle_pos[1] - label.get_size()[1] * int(direction > 0))

                pg.draw.aaline(
                    self.canvas,
                    color,
                    start_pos,
                    end_pos
                )
            else:
                label_pos = (
                    start_pos[0] + 0.5 * self.node_size, start_pos[1] + 0.5 * self.node_size
                )
                pg.draw.arc(
                    self.canvas,
                    color,
                    pg.rect.Rect(
                        start_pos[0], start_pos[1],
                        0.5*self.node_size, 0.5*self.node_size
                    ),
                    0,
                    360
                )

            self.canvas.blit(
                label,
                label_pos
            )

        # update node positions
        for _id, node in self.vertices.items():
            # total attraction
            att_direct = [0.0, 0.0]
            start_pos = node['vis'].rect.center

            for edge in node['in_edges']:
                edge = self.edges[edge]
                end_pos = self.vertices[edge['node1']]['vis'].rect.center

                delta = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                distance = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
                if distance > (
                        self.vertices[edge['node1']]['vis'].radius *
                        (self.init_rad_factor - self.rad_factor) +
                        self.safe_margin
                ):
                    att_direct[0] += (edge['strength'] * delta[0])
                    att_direct[1] += (edge['strength'] * delta[1])

            # total repulsion
            rep_direct = [0.0, 0.0]
            for rid, verx in self.vertices.items():
                if _id != rid:
                    end_pos = verx['vis'].rect.center
                    delta = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
                    distance = (delta[0]**2 + delta[1]**2) ** 0.5
                    delta = self.normalize(delta)
                    effective_rad = verx['vis'].radius * (self.init_rad_factor - self.rad_factor)
                    if distance < effective_rad:
                        rep_direct[0] -= delta[0] / (distance / effective_rad + EPS)
                        rep_direct[1] -= delta[1] / (distance / effective_rad + EPS)

            total_direct = (
                    self.att_factor * att_direct[0] + self.rep_factor * rep_direct[0],
                    self.att_factor * att_direct[1] + self.rep_factor * rep_direct[1]
            )

            shift = self.normalize(total_direct, self.force)
            node['shift'] = shift

        for node in self.vertices.values():
            compensation = (
                self.center[0] - node['vis'].rect.center[0],
                self.center[1] - node['vis'].rect.center[1]
            )
            compensation = self.normalize(compensation, self.gravitation)
            shift = (
                node['shift'][0] + compensation[0],
                node['shift'][1] + compensation[1]
            )
            node['vis'].move(shift)

        self.force *= self.force_decay
        self.gravitation *= self.force_decay
        if self.force <= DUMP_FORCE:
            self.force = 0.0
        if self.gravitation <= DUMP_FORCE:
            self.gravitation = 0
        self.rad_factor *= self.rad_decay

        self.main_group.draw(self.canvas)
        self.main_group.update()

    @staticmethod
    def normalize(x: tuple, s=1.0):
        mag = (x[0]**2 + x[1]**2)**0.5
        return s * x[0]/(mag+EPS), s * x[1]/(mag+EPS)

    def normalize_edges(self):
        # normalize out edge strength
        for node in self.vertices.values():
            max_strength = 1
            for edge in node['out_edges']:
                edge = self.edges[edge]
                strength = sum(edge['actions'].values())
                edge['strength'] = strength
                if strength > max_strength:
                    max_strength = strength

            for edge in node['out_edges']:
                self.edges[edge]['strength'] /= max_strength


class Gridworld:
    def __init__(
            self,
            canvas_size: tuple[int, int],
            cmap: str = 'Pastel1',
            annotate: bool = True
    ):
        self.canvas_size = canvas_size
        self.colors = None
        self.grid_size = None
        self.shape = None
        self.current_position = None

        self.cmap = colormap.cmap_builder(cmap)
        self.annotate = annotate
        self.canvas = Surface(self.canvas_size)
        self.canvas.fill(COLORS['bg'])

    def handle(self, event):
        event_type = event[0]
        if event_type == 'reset':
            self.colors = event[1]['gridworld_map']
        elif event_type == 'new_true_pos':
            self.current_position = event[1]

    def update(self):
        self.canvas.fill(COLORS['bg'])
        if self.colors is not None:
            self.shape = len(self.colors), len(self.colors[0])
            self.grid_size = min(
                self.canvas_size[0] // self.shape[0],
                self.canvas_size[1] // self.shape[1]
            )
            annotation_font = pg.font.SysFont('arial', int(0.5 * self.grid_size))

            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    color = self.colors[row][col]
                    if color < 0:
                        color = COLORS['bg']
                    else:
                        color = convert_color(self.cmap(color))

                    x = col * self.grid_size
                    y = row * self.grid_size
                    pg.draw.rect(self.canvas, color, (x, y, self.grid_size, self.grid_size))

                    if self.annotate:
                        text = annotation_font.render(
                            str(self.colors[row][col]), True, COLORS['text']
                        )
                        self.canvas.blit(text, (x, y))

            if self.current_position is not None:
                text = annotation_font.render(
                    str('A'), True, COLORS['text']
                )
                self.canvas.blit(
                    text,
                    (
                        int(self.current_position[1] * self.grid_size + 0.5 * self.grid_size),
                        int(self.current_position[0] * self.grid_size + 0.5 * self.grid_size)
                    )
                )
