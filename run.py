import os
import subprocess
import time
import sys

# --- Project Details ---
PROJECT_NAME = "Sign Language Interpreter"
DESCRIPTION = "A comprehensive tool for real-time sign detection and text-to-sign/voice-to-sign conversion."
TEAM_MEMBERS = ["Divyansh Singh", "Yash Vardhan Seth", "Arjun Raj", "Mridul Singh", "Yuvraj Singh"]
VERSION = "1.0"
# -----------------------------------------

# --- Style and Colors ---
class Style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def clear_console():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def loading_animation(text, duration):
    """Displays a simple loading animation."""
    print()
    chars = "|/-\\"
    start_time = time.time()
    while time.time() - start_time < duration:
        for char in chars:
            sys.stdout.write(f"\r{Style.YELLOW}{text} {char}{Style.END}")
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write(f"\r{' ' * (len(text) + 2)}\r") # Clear the line

def display_header():
    """Displays the project header."""
    clear_console()
    print(f"{Style.BLUE}{'=' * 60}{Style.END}")
    print(f"{Style.BOLD}{Style.GREEN}{PROJECT_NAME.center(60)}{Style.END}")
    print(f"{Style.YELLOW}{VERSION.center(60)}{Style.END}")
    print(f"{Style.BLUE}{'-' * 60}{Style.END}")
    print(f"{DESCRIPTION}\n")
    print(f"{Style.BOLD}Developed by:{Style.END} {', '.join(TEAM_MEMBERS)}")
    print(f"{Style.BLUE}{'=' * 60}{Style.END}\n")
    loading_animation("Initializing...", 2)

def run_script(command, path, title):
    """Runs a script in its specific directory."""
    clear_console()
    print(f"{Style.GREEN}--- Launching {Style.BOLD}{title}{Style.END}{Style.GREEN} ---{Style.END}\n")
    print(f"Executing command: {Style.YELLOW}{' '.join(command)}{Style.END} in folder: {Style.YELLOW}{path}{Style.END}")
    print("Press CTRL+C to stop the script and return to the menu.\n")
    time.sleep(2)
    
    try:
        # Use subprocess.run to execute the command
        # We pass `cwd` to run the command in the specified directory
        subprocess.run(command, cwd=path, check=True)
    except FileNotFoundError:
        print(f"{Style.RED}Error: The directory '{path}' was not found.{Style.END}")
        print("Please ensure the project structure is correct.")
    except subprocess.CalledProcessError as e:
        print(f"{Style.RED}An error occurred while running the script: {e}{Style.END}")
    except KeyboardInterrupt:
        print(f"\n\n{Style.YELLOW}Returning to the main menu...{Style.END}")
    
    input(f"\n{Style.BLUE}Press Enter to continue...{Style.END}")


# --- Main Application ---
def main():
    """The main menu loop."""
    display_header()
    while True:
        print(f"\n{Style.BOLD}Please choose a mode to run:{Style.END}")
        print("  1. Real-time Sign Detection (from camera)")
        print("  2. Text-to-Sign/Voice-to-Sign Language Conversion")
        print(f"  3. {Style.RED}Exit{Style.END}")

        choice = input(f"\n{Style.GREEN}Enter your choice (1/2/3): {Style.END}")

        if choice == '1':
            # Command: uv run realtime.py
            # Directory: src
            run_script(['uv', 'run', 'realtime.py'], 'src', 'Real-time Sign Detection')
            display_header()
        elif choice == '2':
            # Command: python main.py
            # Directory: Two-Way-Sign-Interpreter
            run_script(['python', 'main.py'], 'Two-Way-Sign-Interpreter', 'Text-to-Sign Translator')
            display_header()
        elif choice == '3':
            print(f"\n{Style.YELLOW}Thank you for using the AI Sign Language Interpreter!{Style.END}")
            break
        else:
            print(f"\n{Style.RED}Invalid choice. Please enter 1, 2, or 3.{Style.END}")
            time.sleep(2)
            display_header()


if __name__ == "__main__":
    main()