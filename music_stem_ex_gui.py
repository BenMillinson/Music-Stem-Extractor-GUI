import os
import torch
import soundfile as sf
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.audio import AudioFile
from tkinter import Tk, Label, Button, filedialog, messagebox, Listbox, Canvas
from tkinterdnd2 import TkinterDnD, DND_FILES
import threading
import pygame
import io

# Initialize pygame mixer for audio playback
pygame.mixer.init()

is_playing = False
current_stem_name = None
loading = False  # Flag to manage loading animation
stems_data = {}  # Dictionary to store stems in memory

def extract_stems(audio_path, model_name="htdemucs"):
    """
    Extracts audio stems from the given audio file and stores them in memory.

    Args:
        audio_path (str): Path to the input audio file.
        model_name (str): The name of the Demucs model to use.

    Returns:
        list: Names of the extracted stems.
    """
    global loading, stems_data
    loading = True
    update_status("Extracting...")

    model = get_model(model_name)
    model.cpu()
    model.eval()

    # Read the audio file
    f = AudioFile(audio_path)
    waveform = f.read(streams=0)
    waveform = waveform[None, ...]

    # Extract base name of the audio file without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    stems_data.clear()  # Clear previously stored stems

    with torch.no_grad():
        sources = apply_model(model, waveform, shifts=1, split=True, overlap=0.25, progress=None)

        for i in range(sources.shape[1]):
            stem = sources[0, i].cpu().numpy().T
            buffer = io.BytesIO()
            sf.write(buffer, stem, model.samplerate, format="WAV")
            buffer.seek(0)  # Reset buffer position for reading
            stem_name = f"{base_name}_stem_{i + 1}"
            stems_data[stem_name] = buffer

    loading = False
    return list(stems_data.keys())

def update_status(message):
    """
    Updates the status label in the GUI.

    Args:
        message (str): The message to display in the status label.
    """
    status_label.config(text=message)

def play_pause_stem():
    """
    Plays or pauses the currently selected stem from the listbox.
    """
    global is_playing, current_stem_name
    selected_index = stems_listbox.curselection()
    if selected_index:
        selected_name = stems_listbox.get(selected_index)

        if not is_playing or current_stem_name != selected_name:
            pygame.mixer.music.stop()
            buffer = stems_data[selected_name]
            buffer.seek(0)
            pygame.mixer.music.load(buffer)
            pygame.mixer.music.play()
            is_playing = True
            play_pause_button.config(text="Pause")
            current_stem_name = selected_name
        elif is_playing:
            pygame.mixer.music.pause()
            is_playing = False
            play_pause_button.config(text="Play")
        else:
            pygame.mixer.music.unpause()
            is_playing = True
            play_pause_button.config(text="Pause")
    else:
        messagebox.showwarning("No Selection", "Please select a stem to play.")

def select_file():
    """
    Opens a file dialog for selecting an audio file.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.flac")])
    if file_path:
        process_file(file_path)

def process_file(file_path):
    """
    Processes the selected audio file and extracts stems.

    Args:
        file_path (str): Path to the selected audio file.
    """
    stems_listbox.delete(0, 'end')
    threading.Thread(target=lambda: display_extracted_stems(file_path)).start()
    start_loading_animation()

def display_extracted_stems(file_path):
    """
    Extracts stems from the given file and displays them in the listbox.

    Args:
        file_path (str): Path to the input audio file.
    """
    stem_names = extract_stems(file_path)
    update_status("Idle")
    if stem_names:
        for name in stem_names:
            stems_listbox.insert('end', name)
        messagebox.showinfo("Separation Complete", "Audio stems have been extracted and are ready to play or download.")
    else:
        messagebox.showerror("Error", "Could not extract stems.")

def start_loading_animation():
    """
    Starts the loading animation.
    """
    canvas.delete("all")
    animate_loading(0)

def animate_loading(angle):
    """
    Animates a circular loading indicator.

    Args:
        angle (int): Current angle of the arc.
    """
    canvas.delete("all")
    if loading:
        x0, y0, x1, y1 = 120, 50, 180, 110
        extent = 90
        canvas.create_arc(x0, y0, x1, y1, start=angle, extent=extent, outline="black", style="arc", width=4)
        root.after(50, animate_loading, (angle + 10) % 360)
    else:
        canvas.delete("all")

def download_stems():
    """
    Downloads the selected stems to a user-specified folder.
    """
    selected_indices = stems_listbox.curselection()
    if selected_indices:
        save_folder = filedialog.askdirectory()
        if save_folder:
            for index in selected_indices:
                stem_name = stems_listbox.get(index)
                try:
                    buffer = stems_data[stem_name]
                    save_path = os.path.join(save_folder, f"{stem_name}.wav")
                    with open(save_path, 'wb') as f:
                        f.write(buffer.read())
                    buffer.seek(0)  # Reset buffer position for future use
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save {stem_name}: {e}")
            messagebox.showinfo("Download Complete", "Selected stems have been saved.")
    else:
        messagebox.showwarning("No Selection", "Please select one or more stems to download.")

def on_file_drop(event):
    """
    Handles drag-and-drop of audio files.

    Args:
        event: The drag-and-drop event.
    """
    file_path = event.data.strip("{}")
    process_file(file_path)

# GUI Setup
root = TkinterDnD.Tk()
root.title("Audio Stem Extractor")
root.geometry("500x600")

Label(root, text="Audio Stem Extractor", font=("Arial", 14)).pack(pady=10)
Button(root, text="Select Audio File", command=select_file).pack(pady=10)

status_label = Label(root, text="Idle", font=("Arial", 12))
status_label.pack()

canvas = Canvas(root, width=300, height=150)
canvas.pack()

Label(root, text="Extracted Stems:", font=("Arial", 12)).pack(pady=5)
stems_listbox = Listbox(root, width=60, height=10, selectmode="multiple")
stems_listbox.pack(pady=10)

play_pause_button = Button(root, text="Play", command=play_pause_stem)
play_pause_button.pack(pady=5)

download_button = Button(root, text="Download Selected", command=download_stems)
download_button.pack(pady=5)

root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', on_file_drop)

root.mainloop()
