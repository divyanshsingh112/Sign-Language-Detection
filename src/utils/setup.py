import json

# Load the configuration file
with open('src\config.json', 'r') as f:
    config = json.load(f)

CLASSES = config['classes']

COLORS = [tuple(color) for color in config['colors']]


def get_classes():
    """Returns the list of class names."""
    return CLASSES

def get_colors():
    """Returns the list of colors as tuples."""
    return COLORS

if __name__ == '__main__':
    print(f"Loaded {len(get_classes())} classes:")
    print(get_classes())
    print(f"\nLoaded {len(get_colors())} colors (as tuples):")
    print(get_colors())