import os
import PIL
from PIL import Image
import speech_recognition as sr
from datetime import datetime


#Path to your FOLDER WITH FULL WORD GIFs 
WORD_GIF_DEST = r"C:\Users\sdivy\OneDrive\Desktop\Sign-Language-Detection\Two-Way-Sign-Interpreter\filtered_data" 
#Path to your FOLDER WITH ALPHABET GIFs
ALPHA_DEST = r"C:\Users\sdivy\OneDrive\Desktop\Sign-Language-Detection\Two-Way-Sign-Interpreter\Alphabet"


#Helper Function
def create_word_gif_map(path):
    """Scans the directory for word GIFs and maps words to their file paths."""
    gif_map = {}
    if not os.path.isdir(path):
        print(f"Warning: Word GIF directory not found at '{path}'")
        return gif_map
        
    for filename in os.listdir(path):
        # Assumes filenames are like 'word.gif' or 'hello there.gif'
        if filename.lower().endswith(('.gif', '.webp')):
            word = os.path.splitext(filename)[0]
            gif_map[word.lower()] = os.path.join(path, filename)
    return gif_map

# Core Function 
def process_gif_frames(gif_path):
    """Opens a GIF and returns a list of its processed frames."""
    frames = []
    try:
        im = Image.open(gif_path)
        for frame_num in range(im.n_frames):
            im.seek(frame_num)
            # Convert frame to a common format and resize
            frame_rgba = im.convert("RGBA")
            frame_rgb = Image.new("RGB", frame_rgba.size, (255, 255, 255))
            frame_rgb.paste(frame_rgba, mask=frame_rgba.split()[3])
            frame_rgb = frame_rgb.resize((380, 260))
            
            # Slow down animation by adding each frame multiple times
            for _ in range(10):
                frames.append(frame_rgb)
    except Exception as e:
        print(f"Error processing GIF at {gif_path}: {e}")
    return frames

def text_to_sign_gif(input_text, word_map):
    """
    Converts a string of text into a GIF, using word GIFs if available.
    """
    print("Generating GIF...")
    all_frames = []
    
    # Process each word
    for word in input_text.lower().split():
        # Check if a GIF for the whole word exists in the map
        if word in word_map:
            print(f"Found word GIF for: '{word}'")
            all_frames.extend(process_gif_frames(word_map[word]))
        else:
            # If not, spell the word out letter by letter
            print(f"Spelling out: '{word}'")
            for char in word:
                letter_gif_path = os.path.join(ALPHA_DEST, f"{char.lower()}_small.gif")
                if os.path.exists(letter_gif_path):
                    all_frames.extend(process_gif_frames(letter_gif_path))
                else:
                    print(f"Warning: No letter GIF found for '{char}'. Skipping.")

    # Save the collected frames as a new GIF
    if all_frames:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"sign_output_{timestamp}.gif"
        
        final_gif = all_frames[0]
        final_gif.save(
            unique_filename,
            save_all=True,
            append_images=all_frames[1:],
            duration=100,
            loop=0
        )
        print(f"\n✅ Successfully generated '{unique_filename}'.")
    else:
        print("\n❌ Could not generate GIF, no valid characters found.")

#Main Application Logic
def main():
    """The main function to run the console application."""
    # Create the word map once at the start
    word_gif_map = create_word_gif_map(WORD_GIF_DEST)
    print(f"Loaded {len(word_gif_map)} word GIFs.")

    while True:
        print("\n--- Console Text-to-Sign Language Translator ---")
        choice = input("Enter '1' to type text or '2' to record voice: ")
        
        final_text = ""
        if choice == '1':
            final_text = input("Enter the text: ")
        elif choice == '2':
            # (Voice recognition code remains the same)
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                print("\n🎤 Listening... Please speak clearly for up to 5 seconds.")
                try:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    print("Recognizing...")
                    final_text = recognizer.recognize_google(audio)
                    print(f"You said: {final_text}")
                except Exception as e:
                    print(f"An error occurred: {e}")
        else:
            print("Invalid choice.")

        if final_text:
            text_to_sign_gif(final_text, word_gif_map)

        another = input("\nDo you want to convert another text? (y/n): ").lower()
        if another != 'y':
            print("Exiting...")
            break

if __name__ == "__main__":
    main()