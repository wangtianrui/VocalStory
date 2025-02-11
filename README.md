# Audiobook Creator

## Overview

Audiobook Creator is an open-source project designed to convert books in various text formats (e.g., EPUB, PDF) into fully voiced audiobooks with intelligent character voice attribution. It leverages modern Natural Language Processing (NLP), Large Language Models (LLMs), and Text-to-Speech (TTS) technologies to create an engaging and dynamic audiobook experience. The project is licensed under the GNU General Public License v3.0 (GPL-3.0), ensuring that it remains free and open for everyone to use, modify, and distribute.

The project consists of three main components:

1. **Text Cleaning and Formatting (`book_to_txt.py`)**:
   - Extracts and cleans text from a book file (e.g., `book.epub`).
   - Normalizes special characters, fixes line breaks, and corrects formatting issues such as unterminated quotes or incomplete lines.
   - Extracts the main content between specified markers (e.g., "PROLOGUE" and "ABOUT THE AUTHOR").
   - Outputs the cleaned text to `converted_book.txt`.

2. **Character Identification and Metadata Generation (`identify_characters_and_output_book_to_jsonl.py`)**:
   - Identifies characters in the text using Named Entity Recognition (NER) with the GLiNER model.
   - Assigns gender and age scores to characters using an LLM via an OpenAI-compatible API.
   - Outputs two files:
     - `speaker_attributed_book.jsonl`: Each line of text annotated with the identified speaker.
     - `character_gender_map.json`: Metadata about characters, including name, age, gender, and gender score.

3. **Audiobook Generation (`generate_audiobook.py`)**:
   - Converts the cleaned text (`converted_book.txt`) or speaker-attributed text (`speaker_attributed_book.jsonl`) into an audiobook using the Kokoro TTS model ([Hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)).
   - Offers two narration modes:
     - **Single-Voice**: Uses a single voice for the entire book.
     - **Multi-Voice**: Assigns different voices to characters based on their gender scores.
   - Saves the audiobook in AAC format to `generated_audiobooks/audiobook.aac`.

## Key Features

- **Multi-Format Support**: Converts books from various formats (EPUB, PDF, etc.) into plain text.
- **Text Cleaning**: Ensures the book text is well-formatted and readable.
- **Character Identification**: Identifies characters and infers their attributes (gender, age) using advanced NLP techniques.
- **Customizable Audiobook Narration**: Supports single-voice or multi-voice narration for enhanced listening experiences.
- **Progress Tracking**: Includes progress bars and execution time measurements for efficient monitoring.
- **Open Source**: Licensed under GPL v3.

## Sample Text and Audio

- `sample_book_and_audio/sample_book.txt`: A sample short story as a starting point.
- `sample_book_and_audio/converted_book.txt`: The cleaned output after text processing.
- `sample_book_and_audio/speaker_attributed_book.jsonl`: The generated speaker-attributed JSONL file.
- `sample_book_and_audio/character_gender_map.json`: The generated character metadata.
- `sample_book_and_audio/sample_multi_voice_audio.aac`: A sample multi-voice audiobook.

## Requirements

- Python 3.12
- Pip 24.0 (MANDATORY)
- Required Python packages:
  - `textract` (For extracting book contents)
  - `openai` (For calling a locally or web-hosted LLM with an OpenAI-compatible endpoint)
  - `requests`
  - `tqdm` (For progress bar updates)
  - `gliner` (For Named Entity Recognition to identify characters in dialogue)
- A locally or web-hosted LLM (e.g., `qwen2.5-14b-instruct-mlx`) with an OpenAI-compatible endpoint.
- Kokoro TTS model hosted via FastAPI (see [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/prakharsr/audiobook-creator.git
   cd audiobook-creator
   ```
2. Create a virtual environment with Python 3.12:
   ```bash
   virtualenv --python="python3.12" .venv
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
4. Install Pip 24.0:
   ```bash
   pip install pip==24.0
   ```
5. Install dependencies (choose CPU or GPU version):
   ```bash
   pip install -r requirements_cpu.txt
   ```
   ```bash
   pip install -r requirements_gpu.txt
   ```
6. Set up your LLM and expose an OpenAI-compatible endpoint (e.g., using LM Studio with `qwen2.5-14b-instruct-mlx`).
7. Set up the Kokoro TTS model. Use CUDA-based GPU inference for faster processing or CPU inference via [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI).
8. Ensure the LLM and TTS endpoints are correctly configured in the project files.

## Usage

1. Activate the virtual environment.
2. Run `python book_to_txt.py` to clean and format the book text. You can manually edit the converted book for fine-grained control. This step requires that you enter your book path in the book_to_txt.py file, for ex. replace `sample_book.txt` with your book path.
3. *(Optional for multi-voice narration)* Run `python identify_characters_and_output_book_to_jsonl.py` to analyze characters and generate metadata. You'll be prompted for a protagonist's name to properly attribute first-person references.
4. Run `python generate_audiobook.py` to generate the audiobook. Choose between single-voice or multi-voice narration.

## Roadmap

Planned future enhancements:

- Use environment variables instead of modifying files directly.
- Improve single-voice narration with distinct dialogue voices.
- Add artwork and chapters, and convert audiobooks to M4B format for better compatibility.

## Support

For issues or questions, open an issue on the [GitHub repository](https://github.com/prakharsr/audiobook-creator/issues).

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please open an issue or pull request to fix a bug or add features.

## Donations

If you find this project useful and would like to support my work, consider donating:  
[PayPal](https://paypal.me/prakharsr)

---

Enjoy creating audiobooks with this project! If you find it helpful, consider giving it a ⭐ on GitHub.