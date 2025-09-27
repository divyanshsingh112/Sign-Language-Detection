import os
import sys
import time
from datetime import datetime
import PIL
import speech_recognition as sr
from PIL import Image

WORD_GIF_DEST = r"C:\Users\sdivy\OneDrive\Desktop\Sign-Language-Detection\Two-Way-Sign-Interpreter\filtered_data"
ALPHA_DEST = r"C:\Users\sdivy\OneDrive\Desktop\Sign-Language-Detection\Two-Way-Sign-Interpreter\Alphabet"

class Style:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def loading_animation(text, duration):
    chars = "⢿⣻⣽⣾⣷⣯⣟⡿"
    start_time = time.time()
    i = 0
    while time.time() - start_time < duration:
        sys.stdout.write(f"\r{Style.YELLOW}{text} {chars[i % len(chars)]}{Style.END}")
        sys.stdout.flush()
        time.sleep(0.05)
        i += 1
    sys.stdout.write(f"\r{' ' * (len(text) + 5)}\r")

def create_word_gif_map(path):
    gif_map = {}
    if not os.path.isdir(path):
        print(f"{Style.RED}Warning: Word GIF directory not found at '{path}'{Style.END}")
        return gif_map
    for filename in os.listdir(path):
        if filename.lower().endswith(('.gif', '.webp')):
            word = os.path.splitext(filename)[0]
            gif_map[word.lower()] = os.path.join(path, filename)
    return gif_map

def process_gif_frames(gif_path):
    frames = []
    try:
        im = Image.open(gif_path)
        for frame_num in range(im.n_frames):
            im.seek(frame_num)
            frame_rgba = im.convert("RGBA")
            frame_rgb = Image.new("RGB", frame_rgba.size, (255, 255, 255))
            frame_rgb.paste(frame_rgba, mask=frame_rgba.split()[3])
            frame_rgb = frame_rgb.resize((380, 260))
            for _ in range(10):
                frames.append(frame_rgb)
    except Exception as e:
        print(f"{Style.RED}Error processing GIF at {gif_path}: {e}{Style.END}")
    return frames

def text_to_sign_gif(input_text, word_map):
    all_frames = []
    print()
    loading_animation("Analyzing text and finding GIFs...", 2)

    for word in input_text.lower().split():
        if word in word_map:
            print(f"  {Style.GREEN}✓ Found word GIF for:{Style.END} '{word}'")
            all_frames.extend(process_gif_frames(word_map[word]))
        else:
            print(f"  {Style.YELLOW}› Spelling out:{Style.END} '{word}'")
            for char in word:
                letter_gif_path = os.path.join(ALPHA_DEST, f"{char.lower()}_small.gif")
                if os.path.exists(letter_gif_path):
                    all_frames.extend(process_gif_frames(letter_gif_path))
                else:
                    print(f"    {Style.RED}✗ No letter GIF for '{char}'. Skipping.{Style.END}")

    if all_frames:
        loading_animation("Generating final GIF file...", 3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"sign_output_{timestamp}.gif"
        final_gif = all_frames[0]
        final_gif.save(
            unique_filename, save_all=True, append_images=all_frames[1:],
            duration=100, loop=0
        )
        print(f"\n{Style.GREEN}╔{'═' * (len(unique_filename) + 18)}╗")
        print(f"║ {Style.BOLD}Successfully generated '{unique_filename}'{Style.END}{Style.GREEN} ║")
        print(f"╚{'═' * (len(unique_filename) + 18)}╝{Style.END}")
    else:
        print(f"\n{Style.RED}Could not generate GIF, no valid characters or words found.{Style.END}")

def main():
    word_gif_map = create_word_gif_map(WORD_GIF_DEST)
    print(f"\n{Style.BLUE}{'=' * 60}{Style.END}")
    print(f"{Style.BOLD}{'Console Text-to-Sign Language Translator'.center(60)}{Style.END}")
    print(f"{Style.BLUE}{'=' * 60}{Style.END}")
    print(f"Loaded {len(word_gif_map)} word GIFs from the database.")

    while True:
        print(f"\n{Style.BOLD}Choose an input method:{Style.END}")
        print("  1. Type text from keyboard")
        print("  2. Record voice from microphone")
        print(f"  3. {Style.RED}Exit{Style.END}")
        
        choice = input(f"\n{Style.GREEN}Enter your choice (1/2/3): {Style.END}")
        
        final_text = ""
        if choice == '1':
            final_text = input(f"{Style.YELLOW}Enter the text: {Style.END}")
        elif choice == '2':
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                print(f"\n{Style.YELLOW}🎤 Listening... Please speak clearly (max 5 seconds).{Style.END}")
                try:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    loading_animation("Recognizing speech...", 2)
                    final_text = recognizer.recognize_google(audio)
                    print(f"{Style.GREEN}You said: {Style.END}{final_text}")
                except Exception as e:
                    print(f"{Style.RED}An error occurred: {e}{Style.END}")
        elif choice == '3':
            print(f"\n{Style.YELLOW}Exiting translator...{Style.END}")
            break
        else:
            print(f"{Style.RED}Invalid choice. Please try again.{Style.END}")

        if final_text:
            text_to_sign_gif(final_text, word_gif_map)
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()