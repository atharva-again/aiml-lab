"""
The Maze Challenge: BFS vs DFS Interactive Visualization
Uses Pygame for smooth, high-fidelity interactive visualization.

Run with: uv run python 0901AI231019_File2.py
"""

import pygame
import sys
from collections import deque

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60
CELL_SIZE = 140

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

CELL_COLORS = {
    "wall": (45, 55, 72),
    "open": (255, 255, 255),
    "start": (72, 187, 120),
    "exit": (245, 101, 101),
    "visited_bfs": (198, 246, 213),
    "visited_dfs": (254, 215, 215),
    "current": (255, 175, 125),
    "path": (237, 137, 54),
}

# Maze Configuration
MAZE_GRID = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
START_POS = (0, 0)
EXIT_POS = (2, 2)


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


def create_maze_graph(grid):
    rows, cols = len(grid), len(grid[0])
    graph = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                        neighbors.append((nr, nc))
                graph[(r, c)] = neighbors
    return graph


class MazeVisualizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Maze Challenge: BFS vs DFS")
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 20)
        self.font_title = pygame.font.SysFont("Arial", 36, bold=True)

        self.graph = create_maze_graph(MAZE_GRID)
        self.reset_state()
        self.setup_ui()

    def reset_state(self):
        self.visited = set()
        self.path = []
        self.current_node = None
        self.traversal_gen = None
        self.last_step_time = 0
        self.is_bfs = True
        self.status = "Select algorithm to start the challenge."
        self.animation_finished = False
        self.comparison_stats = {"bfs": 0, "dfs": 0}

    def setup_ui(self):
        self.buttons = [
            Button(
                (100, 750, 180, 50), "Run BFS", COLOR_SUCCESS, (56, 161, 105), "bfs"
            ),
            Button((300, 750, 180, 50), "Run DFS", COLOR_DANGER, (229, 62, 62), "dfs"),
            Button(
                (500, 750, 180, 50), "Reset", COLOR_TEXT_DIM, (74, 85, 104), "reset"
            ),
        ]
        self.speed_slider = Slider((800, 770, 400, 12), 0.5, 5.0, 1.5)

    def get_bfs_gen(self, start):
        visited = {start}
        queue = deque([(start, [start])])
        yield ("init", start, visited, list(queue))

        while queue:
            node, path = queue.popleft()
            self.current_node = node
            self.path = path
            self.comparison_stats["bfs"] += 1
            yield ("visit", node, visited, path)

            if node == EXIT_POS:
                yield ("done", node, visited, path)
                return

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    yield ("discover", neighbor, visited, path)
        yield ("fail", None, visited, [])

    def get_dfs_gen(self, start):
        visited = set()
        stack = [(start, [start])]

        while stack:
            node, path = stack.pop()
            if node not in visited:
                visited.add(node)
                self.current_node = node
                self.path = path
                self.comparison_stats["dfs"] += 1
                yield ("visit", node, visited, path)

                if node == EXIT_POS:
                    yield ("done", node, visited, path)
                    return

                for neighbor in reversed(self.graph[node]):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
                        yield ("discover", neighbor, visited, path)
        yield ("fail", None, visited, [])

    def draw_maze_grid(self, start_x, start_y):
        for r in range(3):
            for c in range(3):
                pos = (r, c)
                color = CELL_COLORS["open"]
                if MAZE_GRID[r][c] == 1:
                    color = CELL_COLORS["wall"]
                elif pos == START_POS:
                    color = CELL_COLORS["start"]
                elif pos == EXIT_POS:
                    color = CELL_COLORS["exit"]
                elif pos == self.current_node:
                    color = CELL_COLORS["current"]
                elif pos in self.path:
                    color = CELL_COLORS["path"]
                elif pos in self.visited:
                    color = (
                        CELL_COLORS["visited_bfs"]
                        if self.is_bfs
                        else CELL_COLORS["visited_dfs"]
                    )

                rect = pygame.Rect(
                    start_x + c * CELL_SIZE,
                    start_y + r * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (226, 232, 240), rect, 2)

                label = ""
                if pos == START_POS:
                    label = "START"
                elif pos == EXIT_POS:
                    label = "EXIT"
                elif MAZE_GRID[r][c] == 1:
                    label = "WALL"

                if label:
                    txt = self.font_ui.render(
                        label,
                        True,
                        COLOR_TEXT if color == CELL_COLORS["open"] else (255, 255, 255),
                    )
                    self.screen.blit(txt, txt.get_rect(center=rect.center))

    def draw_graph_view(self, start_x, start_y):
        scale = 160
        # Edges
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                p1 = (start_x + node[1] * scale, start_y + node[0] * scale)
                p2 = (start_x + neighbor[1] * scale, start_y + neighbor[0] * scale)
                pygame.draw.line(self.screen, (160, 174, 192), p1, p2, 3)

        # Nodes
        for node in self.graph.keys():
            pos = (start_x + node[1] * scale, start_y + node[0] * scale)
            color = CELL_COLORS["open"]
            if node == self.current_node:
                color = CELL_COLORS["current"]
            elif node in self.path:
                color = CELL_COLORS["path"]
            elif node in self.visited:
                color = (
                    CELL_COLORS["visited_bfs"]
                    if self.is_bfs
                    else CELL_COLORS["visited_dfs"]
                )

            pygame.draw.circle(self.screen, color, pos, 35)
            pygame.draw.circle(self.screen, (45, 55, 72), pos, 35, 3)

            txt = self.font_ui.render(str(node), True, COLOR_TEXT)
            self.screen.blit(txt, txt.get_rect(center=pos))

    def draw(self):
        self.screen.fill(COLOR_BG)

        # Header
        pygame.draw.rect(self.screen, COLOR_PANEL, (0, 0, WINDOW_WIDTH, 80))
        title = self.font_title.render(
            "The Maze Challenge: Pathfinding Visualizer", True, COLOR_TEXT
        )
        self.screen.blit(title, (50, 15))

        # Split View
        self.draw_maze_grid(150, 180)
        self.draw_graph_view(850, 220)

        # Captions
        c1 = self.font_main.render("Maze Layout", True, COLOR_TEXT)
        self.screen.blit(c1, (150, 130))
        c2 = self.font_main.render("Graph Logical Representation", True, COLOR_TEXT)
        self.screen.blit(c2, (800, 130))

        # Bottom Panel
        pygame.draw.rect(self.screen, COLOR_PANEL, (0, 700, WINDOW_WIDTH, 200))
        pygame.draw.line(self.screen, (226, 232, 240), (0, 700), (WINDOW_WIDTH, 700), 2)

        status_txt = self.font_ui.render(f"Status: {self.status}", True, COLOR_TEXT)
        self.screen.blit(status_txt, (100, 715))

        # Legend
        lx = 100
        for label, color in [
            ("Visited", (198, 246, 213)),
            ("Current", (255, 175, 125)),
            ("Path", (237, 137, 54)),
        ]:
            pygame.draw.rect(self.screen, color, (lx, 820, 20, 20))
            t = self.font_ui.render(label, True, COLOR_TEXT)
            self.screen.blit(t, (lx + 30, 820))
            lx += 150

        # Stats
        stats_txt = f"Nodes Visited - BFS: {self.comparison_stats['bfs']} | DFS: {self.comparison_stats['dfs']}"
        st_surf = self.font_ui.render(stats_txt, True, COLOR_ACCENT)
        self.screen.blit(st_surf, (800, 715))

        speed_txt = self.font_ui.render(
            f"Simulation Speed: {self.speed_slider.value:.1f}x", True, COLOR_TEXT
        )
        self.screen.blit(speed_txt, (800, 800))

        for btn in self.buttons:
            btn.draw(self.screen)
        self.speed_slider.draw(self.screen)

        pygame.display.flip()

    def update(self):
        if self.traversal_gen and not self.animation_finished:
            now = pygame.time.get_ticks()
            delay = 1000 / self.speed_slider.value
            if now - self.last_step_time > delay:
                try:
                    res = next(self.traversal_gen)
                    type, node, visited, path = res
                    self.visited = visited
                    self.last_step_time = now

                    if type == "visit":
                        self.status = f"Evaluating cell {node}..."
                    elif type == "done":
                        self.status = "Exit Found! Path highlighted."
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
                            self.traversal_gen = self.get_bfs_gen(START_POS)
                        elif btn.action_id == "dfs":
                            self.reset_state()
                            self.is_bfs = False
                            self.traversal_gen = self.get_dfs_gen(START_POS)
                        elif btn.action_id == "reset":
                            self.reset_state()

                self.speed_slider.handle_event(event)

            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    MazeVisualizer().run()
