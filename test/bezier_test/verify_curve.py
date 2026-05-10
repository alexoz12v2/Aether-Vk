from PIL import Image
import sys

def check_connected_components(image_path):
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    pixels = img.load()
    
    # Find all yellow pixels
    yellow_pixels = set()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r > 100 and g > 100 and b < 50:  # Yellowish
                yellow_pixels.add((x, y))
                
    if not yellow_pixels:
        print(f"No yellow pixels found in {image_path}!")
        return False
        
    print(f"Found {len(yellow_pixels)} yellow pixels.")
    
    # BFS to find connected components
    visited = set()
    components = 0
    
    for start_pixel in yellow_pixels:
        if start_pixel not in visited:
            components += 1
            # Run BFS
            queue = [start_pixel]
            visited.add(start_pixel)
            
            while queue:
                cx, cy = queue.pop(0)
                # Check 8 neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in yellow_pixels and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            
    print(f"Found {components} connected component(s).")
    if components == 1:
        print("Success: The curve is one continuous piece!")
        return True
    else:
        print(f"Error: The curve is broken into {components} disconnected pieces!")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_curve.py <image.png>")
        sys.exit(1)
        
    success = check_connected_components(sys.argv[1])
    if not success:
        sys.exit(1)
