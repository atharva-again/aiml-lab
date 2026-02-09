"""
Graph Traversal: Interactive BFS and DFS Visualization
Uses Pygame for smooth, high-fidelity interactive visualization.

Run with: uv run python 0901AI231019_File1.py
"""

import pygame
import sys
import math
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 850
FPS = 60

# Professional Color Palette
COLOR_BG = (245, 247, 250)
COLOR_PANEL = (255, 255, 255)
COLOR_TEXT = (45, 55, 72)
COLOR_TEXT_DIM = (113, 128, 150)
COLOR_PRIMARY = (66, 153, 225)
COLOR_SUCCESS = (72, 187, 120)
COLOR_WARNING = (237, 137, 54)
COLOR_DANGER = (245, 101, 101)
COLOR_ACCENT = (159, 122, 234)

NODE_COLORS = {
    "unvisited": (237, 242, 247),
    "queued": (254, 235, 200),
    "visited": (198, 246, 213),
    "current": (255, 175, 125),
    "border_unvisited": (203, 213, 224),
    "border_queued": (237, 137, 54),
    "border_visited": (56, 161, 105),
    "border_current": (221, 107, 32),
}

# Graph Configuration
GRAPH = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": ["F"], "F": []}

NODE_POSITIONS = {
    "A": (600, 150),
    "B": (350, 350),
    "C": (850, 350),
    "D": (150, 550),
    "E": (450, 550),
    "F": (850, 550),
}

NODE_RADIUS = 42


class UIComponent:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect)
        self.hovered = False

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered


class Button(UIComponent):
    def __init__(self, rect, text, color, hover_color, action_id):
        super().__init__(rect)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.action_id = action_id
        self.font = pygame.font.SysFont("Arial", 22, bold=True)

    def draw(self, screen):
        draw_color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(screen, draw_color, self.rect, border_radius=10)

        text_surf = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)


class RadioButton(UIComponent):
    def __init__(self, x, y, label, node_id, group):
        super().__init__((x, y, 110, 40))
        self.label = label
        self.node_id = node_id
        self.group = group
        self.selected = False
        self.font = pygame.font.SysFont("Arial", 18)

    def draw(self, screen):
        center_y = self.rect.centery
        circle_x = self.rect.x + 20
        pygame.draw.circle(screen, COLOR_TEXT_DIM, (circle_x, center_y), 10, 2)
        if self.selected:
            pygame.draw.circle(screen, COLOR_PRIMARY, (circle_x, center_y), 6)

        text_surf = self.font.render(self.label, True, COLOR_TEXT)
        screen.blit(text_surf, (circle_x + 18, center_y - 10))


class Slider(UIComponent):
    def __init__(self, rect, min_val, max_val, initial_val):
        super().__init__(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.handle_rect = pygame.Rect(0, 0, 22, 22)
        self.dragging = False

    def get_handle_pos(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + ratio * self.rect.width

    def draw(self, screen):
        pygame.draw.rect(screen, (226, 232, 240), self.rect, border_radius=5)
        handle_x = self.get_handle_pos()
        progress_rect = pygame.Rect(
            self.rect.x, self.rect.y, handle_x - self.rect.x, self.rect.height
        )
        pygame.draw.rect(screen, COLOR_ACCENT, progress_rect, border_radius=5)

        self.handle_rect.center = (handle_x, self.rect.centery)
        pygame.draw.circle(screen, (255, 255, 255), self.handle_rect.center, 11)
        pygame.draw.circle(screen, COLOR_ACCENT, self.handle_rect.center, 11, 2)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos) or self.rect.collidepoint(
                event.pos
            ):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            rel_x = max(0, min(event.pos[0] - self.rect.x, self.rect.width))
            ratio = rel_x / self.rect.width
            self.value = self.min_val + ratio * (self.max_val - self.min_val)


class GraphVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Graph Traversal: BFS and DFS Animation")
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 20)

        self.reset_state()
        self.setup_ui()

    def reset_state(self):
        self.visited = set()
        self.order = []
        self.active_structure = []
        self.current_node = None
        self.traversal_gen = None
        self.last_step_time = 0
        self.is_bfs = True
        self.start_node = "A"
        self.status = "System Ready. Select start node and algorithm."
        self.animation_finished = False

    def setup_ui(self):
        self.buttons = [
            Button((50, 740, 150, 45), "Run BFS", COLOR_SUCCESS, (56, 161, 105), "bfs"),
            Button(
                (210, 740, 150, 45), "Run DFS", COLOR_PRIMARY, (49, 130, 206), "dfs"
            ),
            Button((370, 740, 150, 45), "Reset", COLOR_DANGER, (229, 62, 62), "reset"),
        ]

        self.radio_group = []
        nodes = sorted(GRAPH.keys())
        for i, node in enumerate(nodes):
            rb = RadioButton(
                750 + (i % 3) * 110,
                710 + (i // 3) * 45,
                f"Node {node}",
                node,
                self.radio_group,
            )
            if node == "A":
                rb.selected = True
            self.radio_group.append(rb)

        self.speed_slider = Slider((750, 815, 300, 10), 0.5, 5.0, 1.0)

    def get_bfs_gen(self, start):
        visited = set()
        queue = deque([start])
        visited.add(start)
        yield ("init", None, visited, list(queue))

        while queue:
            node = queue.popleft()
            self.order.append(node)
            yield ("visit", node, visited, list(queue))

            for neighbor in GRAPH[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    yield ("queue", neighbor, visited, list(queue))
        yield ("done", None, visited, [])

    def get_dfs_gen(self, start):
        visited = set()
        stack = [start]
        yield ("init", None, visited, list(stack))

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                self.order.append(node)
                yield ("visit", node, visited, list(stack))

                for neighbor in reversed(GRAPH[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        yield ("stack", neighbor, visited, list(stack))
        yield ("done", None, visited, [])

    def draw_arrow(self, start_pos, end_pos, color):
        pygame.draw.line(self.screen, color, start_pos, end_pos, 3)
        angle = math.atan2(start_pos[1] - end_pos[1], start_pos[0] - end_pos[0])
        dist = 16
        offset_angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        tip_x = end_pos[0] - (NODE_RADIUS + 2) * math.cos(offset_angle)
        tip_y = end_pos[1] - (NODE_RADIUS + 2) * math.sin(offset_angle)

        p1 = (
            tip_x + dist * math.cos(angle + 0.4),
            tip_y + dist * math.sin(angle + 0.4),
        )
        p2 = (
            tip_x + dist * math.cos(angle - 0.4),
            tip_y + dist * math.sin(angle - 0.4),
        )
        pygame.draw.polygon(self.screen, color, [(tip_x, tip_y), p1, p2])

    def draw(self):
        self.screen.fill(COLOR_BG)

        # Title
        title = self.font_title.render("Graph Traversal Visualizer", True, COLOR_TEXT)
        self.screen.blit(title, (50, 25))

        # Edges
        for node, neighbors in GRAPH.items():
            for neighbor in neighbors:
                self.draw_arrow(
                    NODE_POSITIONS[node], NODE_POSITIONS[neighbor], (160, 174, 192)
                )

        # Nodes
        for node, pos in NODE_POSITIONS.items():
            state = "unvisited"
            if node == self.current_node:
                state = "current"
            elif node in self.order:
                state = "visited"
            elif node in self.active_structure:
                state = "queued"

            pygame.draw.circle(self.screen, NODE_COLORS[state], pos, NODE_RADIUS)
            pygame.draw.circle(
                self.screen, NODE_COLORS["border_" + state], pos, NODE_RADIUS, 4
            )

            label = self.font_main.render(node, True, COLOR_TEXT)
            self.screen.blit(label, label.get_rect(center=pos))

            if node in self.order:
                idx = self.order.index(node) + 1
                s_font = pygame.font.SysFont("Arial", 16, bold=True)
                step_surf = s_font.render(str(idx), True, (255, 255, 255))
                pygame.draw.circle(
                    self.screen, COLOR_ACCENT, (pos[0] + 32, pos[1] - 32), 12
                )
                self.screen.blit(
                    step_surf, step_surf.get_rect(center=(pos[0] + 32, pos[1] - 32))
                )

        # Panel
        pygame.draw.rect(self.screen, COLOR_PANEL, (0, 680, WINDOW_WIDTH, 170))
        pygame.draw.line(self.screen, (226, 232, 240), (0, 680), (WINDOW_WIDTH, 680), 2)

        # Status
        status_surf = self.font_ui.render(self.status, True, COLOR_TEXT)
        self.screen.blit(status_surf, (50, 695))

        # Order
        order_text = "Path: " + " -> ".join(self.order)
        order_surf = self.font_ui.render(order_text, True, COLOR_TEXT_DIM)
        self.screen.blit(order_surf, (50, 795))

        # Structure
        struct_name = "Queue (FIFO)" if self.is_bfs else "Stack (LIFO)"
        struct_text = f"{struct_name}: {self.active_structure}"
        struct_surf = self.font_ui.render(struct_text, True, COLOR_ACCENT)
        self.screen.blit(struct_surf, (550, 695))

        for btn in self.buttons:
            btn.draw(self.screen)
        for rb in self.radio_group:
            rb.draw(self.screen)
        self.speed_slider.draw(self.screen)

        speed_label = self.font_ui.render(
            f"Speed: {self.speed_slider.value:.1f}x", True, COLOR_TEXT
        )
        self.screen.blit(speed_label, (750, 785))

        pygame.display.flip()

    def update(self):
        if self.traversal_gen and not self.animation_finished:
            now = pygame.time.get_ticks()
            delay = 1000 / self.speed_slider.value
            if now - self.last_step_time > delay:
                try:
                    step_type, node, visited_set, structure = next(self.traversal_gen)
                    self.active_structure = structure
                    self.last_step_time = now

                    if step_type == "visit":
                        self.current_node = node
                        self.status = f"Exploring node {node}..."
                    elif step_type in ["queue", "stack"]:
                        self.status = f"Discovered node {node}."
                    elif step_type == "done":
                        self.status = "Traversal complete."
                        self.current_node = None
                        self.animation_finished = True
                except StopIteration:
                    self.animation_finished = True

    def run(self):
        running = True
        while running:
            mx, my = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                for btn in self.buttons:
                    btn.check_hover((mx, my))
                    if event.type == pygame.MOUSEBUTTONDOWN and btn.hovered:
                        if btn.action_id == "bfs":
                            self.reset_state()
                            self.is_bfs = True
                            self.traversal_gen = self.get_bfs_gen(self.start_node)
                        elif btn.action_id == "dfs":
                            self.reset_state()
                            self.is_bfs = False
                            self.traversal_gen = self.get_dfs_gen(self.start_node)
                        elif btn.action_id == "reset":
                            self.reset_state()

                for rb in self.radio_group:
                    if event.type == pygame.MOUSEBUTTONDOWN and rb.rect.collidepoint(
                        (mx, my)
                    ):
                        for b in self.radio_group:
                            b.selected = False
                        rb.selected = True
                        self.start_node = rb.node_id
                        self.reset_state()

                self.speed_slider.handle_event(event)

            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    GraphVisualizer().run()
