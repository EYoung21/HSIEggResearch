#!/usr/bin/env python3
"""
Script to fix Unicode encoding issues in G4_model.py for Windows compatibility
"""

def fix_unicode_in_file(filename):
    """Replace Unicode emojis with ASCII equivalents"""
    
    unicode_replacements = {
        '📂': '[LOAD]',
        '🔍': '[SEARCH]', 
        '✅': '[OK]',
        '🔄': '[PROC]',
        '🚀': '[START]',
        '📊': '[DATA]',
        '💾': '[SAVE]',
        '🎯': '[RESULT]',
        '🎉': '[SUCCESS]',
        '❌': '[ERROR]',
        '🏆': '[BEST]',
        '⚙️': '[CONFIG]',
        '🧹': '[CLEANUP]',
        '🌊': '[POOL]',
        '📥': '[INPUT]',
        '🔧': '[BUILD]',
        '📍': '[POS]',
        '🏗️': '[ARCH]',
        '📚': '[MODEL]',
        '📂': '[FOLD]',
        '🔹': '•',  # Blue diamond bullet
        '\U0001f539': '•',  # Unicode bullet point
        '▪': '*',   # Small black square
        '▫': '*',   # Small white square
        '◦': '*',   # Bullet operator
        '•': '*'    # Bullet point
    }
    
    print(f"Fixing Unicode characters in {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Unicode characters
    for unicode_char, ascii_replacement in unicode_replacements.items():
        content = content.replace(unicode_char, ascii_replacement)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {len(unicode_replacements)} Unicode character types in {filename}")

if __name__ == "__main__":
    fix_unicode_in_file('G4_model.py')
    print("Unicode fix completed!") 