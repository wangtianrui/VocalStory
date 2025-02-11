"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
import textract

def fix_unterminated_quotes(text: str):
    """Fixes unterminated quotes in the given text."""
    lines = text.splitlines()  # Split the text into lines
    fixed_text = []
    in_quote = False  # Track if we're inside a quote

    for line in lines:
        quote_indices = [i for i, char in enumerate(line) if char == '"']  # Find all quote positions
        if not quote_indices:
            fixed_text.append(line)  # No quotes in this line, add as is
            continue

        # Process quotes in the line
        new_line = ""
        for i, char in enumerate(line):
            if char == '"':
                in_quote = not in_quote  # Toggle quote state
            new_line += char

        # If the line ends with an unterminated quote, add a closing quote
        if in_quote and not line.endswith('"'):
            new_line += '"'
            in_quote = False  # Reset quote state

        # If the line ends with an unterminated quote, add a closing quote
        if in_quote and not new_line.startswith('"'):
            new_line = '"' + new_line
            in_quote = False  # Reset quote state

        fixed_text.append(new_line)

    return "\n".join(fixed_text)  # Join lines back into a single string

def extract_main_content(text, start_marker="PROLOGUE", end_marker="ABOUT THE AUTHOR"):
    """
    Extracts the main content of a book between two markers (case-insensitive).
    Handles edge cases such as multiple marker occurrences and proper content boundaries.
    
    Args:
        text (str): The full text of the book.
        start_marker (str): The marker indicating the start of the main content.
        end_marker (str): The marker indicating the end of the main content.
    
    Returns:
        str: The extracted main content.
        
    Raises:
        ValueError: If markers are not found or if their positions are invalid.
    """
    
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        if not start_marker or not end_marker:
            raise ValueError("Markers must be non-empty strings")
        
        # Find all occurrences of markers
        start_positions = []
        end_positions = []
        pos = 0
        
        # Find all start marker positions
        while True:
            pos = text.find(start_marker, pos)
            if pos == -1:
                break
            start_positions.append(pos)
            pos += 1
            
        # Find all end marker positions
        pos = 0
        while True:
            pos = text.find(end_marker, pos)
            if pos == -1:
                break
            end_positions.append(pos)
            pos += 1
            
        # Validate marker existence
        if not start_positions:
            raise ValueError(f"Start marker '{start_marker}' not found in the text")
        if not end_positions:
            raise ValueError(f"End marker '{end_marker}' not found in the text")
            
        # Find the correct pair of markers
        start_index = start_positions[len(start_positions)-1]
        end_index = end_positions[len(end_positions)-1]
    
        if start_index is None or end_index is None:
            raise ValueError("Could not find valid marker positions with substantial content between them")
            
        # Extract and clean the content
        main_content = text[start_index:end_index].strip()
        
        # Validate extracted content
        if len(main_content) < 100:  # Adjust this threshold as needed
            raise ValueError("Extracted content is suspiciously short")
            
        # Remove any leading/trailing chapter markers or section headers
        lines = main_content.split('\n')
        while lines and (
            any(marker.lower() in lines[0].lower() 
                for marker in [start_marker, end_marker, 'chapter', 'part', 'book'])):
            lines.pop(0)
        while lines and (
            any(marker.lower() in lines[-1].lower() 
                for marker in [start_marker, end_marker, 'chapter', 'part', 'book'])):
            lines.pop()
            
        return '\n'.join(lines).strip()
    except Exception as e:
        print("Error", e, ", not extracting main content.")
        return text
    
def normalize_line_breaks(text):
    # Split the text into lines
    lines = text.splitlines()
    
    # Filter out empty lines and strip any leading/trailing whitespace
    non_empty_lines = [line.strip() for line in lines if line.strip()]
    
    # Join the lines with a single line break
    normalized_text = '\n'.join(non_empty_lines)
    
    return normalized_text

def fix_unterminated_text(input_text):
    lines = input_text.split('\n')
    fixed_lines = []
    
    for line in lines:
        line = line.strip()
        
        # If it's a header or all caps, or it contains only numbers, keep it separate
        if line.isupper() or line.isdigit():
            fixed_lines.append(line)
            continue
        
        # Merge with the previous line if it doesn't end in punctuation
        if fixed_lines and not fixed_lines[-1].endswith(('.', '!', '?', '"')): 
            fixed_lines[-1] += ' ' + line
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

text: str = textract.process('sample_book.txt', encoding='utf-8').decode() # decode using textract

text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("\u2018", "'") # Normalize text by replacing curly quotes and apostrophes with standard ASCII equivalents

text = normalize_line_breaks(text) # Remove multiple line breaks, normalize it 

# optional, do it for books which are not formatted properly and have unterminated lines
# text = fix_unterminated_text(text)

text = fix_unterminated_quotes(text) # Fix missing quotes in a dialogue, works both ways for opening and closing quotes

# optional, you can do this step manually by editing book.txt also or you can use this function
# text = extract_main_content(text, start_marker="PROLOGUE", end_marker="ABOUT THE AUTHOR") 

with open("converted_book.txt", 'w', encoding='utf-8') as fout:
    fout.write(text)