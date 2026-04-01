import pygame
import numpy as np

BLACK = (0, 0, 0)

class Button:
    def __init__(self, x, y, w, h, text, font, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.callback = callback
        self.color = (200, 200, 200)
        self.text_surf = self.font.render(self.text, True, BLACK)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.text_surf, self.text_rect)

class DebugGUI:
    def __init__(self, screen, font, start_x, start_y=10):
        self.screen = screen
        self.font = font
        self.x = start_x
        self.y = start_y
        self.line_height = font.get_height() + 2

    def _format_np(self, item):
        if isinstance(item, np.ndarray):
            if item.ndim == 1:
                return f"[{', '.join(f'{x:8.2f}' for x in item)}]"
            else:
                return str(item).replace("\n", " ")
        if isinstance(item, float):
            return f"{item:8.2f}"
        return str(item)

    def draw(self, title, data, color):
        title_surf = self.font.render(title, True, BLACK)
        self.screen.blit(title_surf, (self.x + 5, self.y))
        self.y += self.line_height * 1.5
        for key, value in data.items():
            text = f"{key:<22}: {self._format_np(value)}"
            text_surf = self.font.render(text, True, color)
            self.screen.blit(text_surf, (self.x + 10, self.y))
            self.y += self.line_height
        self.y += self.line_height


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val, label="Speed"):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val, self.max_val = min_val, max_val
        self.val = start_val
        self.label = label
        self.dragging = False
        self.handle_w = 10
        self.update_handle_pos()

    def update_handle_pos(self):
        percent = (self.val - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.x + percent * (self.rect.w - self.handle_w)
        self.handle_rect = pygame.Rect(
            handle_x, self.rect.y, self.handle_w, self.rect.h
        )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.val = self.min_val + (event.pos[0] - self.rect.x) / self.rect.w * (
                self.max_val - self.min_val
            )
            self.val = max(self.min_val, min(self.max_val, self.val))
            self.update_handle_pos()

    def get_value(self):
        return int(self.val)

    def draw(self, screen, font):
        pygame.draw.rect(screen, (200, 200, 200), self.rect)
        pygame.draw.rect(screen, (100, 100, 100), self.handle_rect)
        text_surf = font.render(f"{self.label}: {self.get_value()}", True, BLACK)
        screen.blit(text_surf, (self.rect.right + 10, self.rect.y))
