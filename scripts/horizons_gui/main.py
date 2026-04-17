import pygame
import urllib.request
import urllib.parse
import json
import sys
import ssl

pygame.init()

# Setup display
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Horizons API GUI App")

font = pygame.font.SysFont(None, 24)
small_font = pygame.font.SysFont(None, 20)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)
BLUE = (100, 100, 255)
YELLOW = (255, 255, 200)

class TextInput:
    def __init__(self, x, y, w, h, label, default_text=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.text = default_text
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # Add unicode only if it's printable
                if event.unicode.isprintable():
                    self.text += event.unicode

    def draw(self, surface):
        color = BLUE if self.active else GRAY
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, color, self.rect, 2)
        txt_surface = font.render(self.text, True, BLACK)
        surface.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))
        lbl_surface = font.render(self.label, True, BLACK)
        surface.blit(lbl_surface, (self.rect.x, self.rect.y - 20))


class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.color = GRAY
        self.hover_color = LIGHT_GRAY

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.callback()

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 1)
        txt_surface = font.render(self.text, True, BLACK)
        surface.blit(txt_surface, (self.rect.x + (self.rect.w - txt_surface.get_width())//2, self.rect.y + 6))


fetched_data = []
headers = []
status_msg = ""
scroll_y = 0


def fetch_data(cmd, start, stop, step):
    global fetched_data, headers, status_msg, scroll_y
    status_msg = "Fetching... (UI will freeze briefly)"
    fetched_data = []
    headers = []
    scroll_y = 0
    
    try:
        base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
        params = {
            "format": "json",
            "COMMAND": f"'{cmd}'",
            "MAKE_EPHEM": "'YES'",
            "EPHEM_TYPE": "'OBSERVER'",
            "START_TIME": f"'{start}'",
            "STOP_TIME": f"'{stop}'",
            "STEP_SIZE": f"'{step}'",
            "QUANTITIES": "'1,9,20'",
            "CSV_FORMAT": "'YES'"
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{base_url}?{query_string}"
        print(f"--- HTTP REQUEST ---\nGET {url}\n")
        
        req = urllib.request.Request(url)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx) as response:
            raw_response = response.read().decode()
            print(f"--- HTTP RESPONSE HEADERS ---\n{response.info()}")
            print(f"--- HTTP RESPONSE BODY (first 500 chars) ---\n{raw_response[:500]}...\n")
            res_data = json.loads(raw_response)
            
        result_text = res_data.get("result", "")
        lines = result_text.split('\n')
        
        in_data = False
        data_lines = []
        header_line = ""
        
        # Parse output markers
        for i, line in enumerate(lines):
            line = line.strip()
            if line == '$$SOE':
                in_data = True
                # The line two lines before $$SOE contains the CSV headers
                if i > 1:
                    header_line = lines[i-2].strip()
                continue
            if line == '$$EOE':
                in_data = False
                break
            if in_data:
                data_lines.append(line)
                
        if header_line:
            headers = [h.strip() for h in header_line.split(',')]
            
        for dl in data_lines:
            row = [v.strip() for v in dl.split(',')]
            fetched_data.append(row)
            
        status_msg = f"Fetched {len(fetched_data)} rows."
    except Exception as e:
        status_msg = f"Error: {e}"

def main():
    global scroll_y, status_msg
    clock = pygame.time.Clock()

    inputs = [
        TextInput(20, 50, 100, 30, "Target Body", "499"),
        TextInput(150, 50, 120, 30, "Start Time", "2024-01-01"),
        TextInput(300, 50, 120, 30, "Stop Time", "2024-01-05"),
        TextInput(450, 50, 100, 30, "Step Size", "1d"),
    ]

    def set_preset(body, start, stop, step):
        inputs[0].text = body
        inputs[1].text = start
        inputs[2].text = stop
        inputs[3].text = step

    presets = [
        Button(20, 5, 80, 20, "Mars", lambda: set_preset("499", "2024-01-01", "2024-01-05", "1d")),
        Button(110, 5, 180, 20, "67P/Churyumov", lambda: set_preset("90000033", "2014-10-30", "2014-11-05", "1d")),
    ]

    def on_fetch():
        fetch_data(inputs[0].text, inputs[1].text, inputs[2].text, inputs[3].text)

    btn = Button(580, 50, 100, 30, "Fetch", on_fetch)

    running = True
    while running:
        screen.fill(WHITE)
        
        start_x = 20
        header_y = 100
        start_y = 125 + scroll_y
        row_height = 25
        col_width = 110
        
        mouse_pos = pygame.mouse.get_pos()
        tooltip_info = None

        if headers and fetched_data:
            # Draw Data Grid first
            y_offset = start_y
            for i, row in enumerate(fetched_data):
                if y_offset + row_height > 125 and y_offset < HEIGHT:
                    for j, val in enumerate(row):
                        rect = pygame.Rect(start_x + j*col_width, y_offset, col_width, row_height)
                        pygame.draw.rect(screen, WHITE, rect)
                        pygame.draw.rect(screen, BLACK, rect, 1)
                        txt = small_font.render(val[:15], True, BLACK)
                        screen.blit(txt, (rect.x + 2, rect.y + 5))
                        
                        # Store tooltip if mouse is hovering over the cell
                        if rect.collidepoint(mouse_pos) and rect.top >= 125:
                            head = headers[j] if j < len(headers) else "Unknown"
                            tooltip_info = (head, val)
                y_offset += row_height

        # Top UI Panel - redraw to cover any scrolled data beneath
        pygame.draw.rect(screen, WHITE, (0, 0, WIDTH, 125))
        
        # Draw headers FIXED to the top panel
        if headers:
            for j, h in enumerate(headers):
                if not h: continue
                rect = pygame.Rect(start_x + j*col_width, header_y, col_width, row_height)
                pygame.draw.rect(screen, LIGHT_GRAY, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
                txt = small_font.render(h[:15], True, BLACK)
                screen.blit(txt, (rect.x + 2, rect.y + 5))

        pygame.draw.line(screen, BLACK, (0, 125), (WIDTH, 125), 2)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEWHEEL:
                scroll_y += event.y * 20
                if scroll_y > 0: scroll_y = 0
            for inp in inputs:
                inp.handle_event(event)
            for p_btn in presets:
                p_btn.handle_event(event)
            btn.handle_event(event)
            
        for inp in inputs:
            inp.draw(screen)
        for p_btn in presets:
            p_btn.draw(screen)
        btn.draw(screen)
        
        status_surf = font.render(status_msg, True, BLACK)
        screen.blit(status_surf, (700, 55))

        # Draw tooltip rendering on top of everything
        if tooltip_info:
            tt_text = f"{tooltip_info[0]}: {tooltip_info[1]}"
            tt_surf = font.render(tt_text, True, BLACK)
            tt_rect = tt_surf.get_rect()
            
            # Constrain tooltip to remain within the screen boundaries
            tt_x = mouse_pos[0] + 15
            tt_y = mouse_pos[1] + 15
            if tt_x + tt_rect.width + 10 > WIDTH:
                tt_x = WIDTH - tt_rect.width - 10
            if tt_y + tt_rect.height + 10 > HEIGHT:
                tt_y = HEIGHT - tt_rect.height - 10
                
            tt_rect.topleft = (tt_x, tt_y)
            
            pygame.draw.rect(screen, YELLOW, tt_rect.inflate(10, 10))
            pygame.draw.rect(screen, BLACK, tt_rect.inflate(10, 10), 1)
            screen.blit(tt_surf, tt_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
