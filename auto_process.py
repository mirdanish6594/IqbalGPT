from deep_translator import GoogleTranslator
import time

def built_in_transliterate(urdu_string):
    """
    A simple, dependency-free function to transliterate Urdu script to Roman Urdu.
    This is a basic implementation and may not be perfect for all words.
    """
    mapping = {
        'ا': 'a', 'آ': 'a', 'ب': 'b', 'پ': 'p', 'ت': 't', 'ٹ': 't', 'ث': 's',
        'ج': 'j', 'چ': 'ch', 'ح': 'h', 'خ': 'kh', 'د': 'd', 'ڈ': 'd', 'ذ': 'z',
        'ر': 'r', 'ڑ': 'r', 'ز': 'z', 'ژ': 'zh', 'س': 's', 'ش': 'sh', 'ص': 's',
        'ض': 'z', 'ط': 't', 'ظ': 'z', 'ع': 'a', 'غ': 'gh', 'ف': 'f', 'ق': 'q',
        'ک': 'k', 'گ': 'g', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ں': 'n', 'و': 'o',
        'ہ': 'h', 'ھ': 'h', 'ء': "'", 'ی': 'i', 'ے': 'e', ' ': ' ', '\n': '\n',
        '۱': '1', '۲': '2', '۳': '3', '۴': '4', '۵': '5', '۶': '6', '۷': '7',
        '۸': '8', '۹': '9', '۰': '0', '؟': '?', '۔': '.', '،': ',',
    }
    
    roman_text = ""
    for char in urdu_string:
        roman_text += mapping.get(char, '') # Get the roman equivalent, or empty string if not found
        
    return roman_text

def auto_process_urdu_text():
    """
    Reads an unstructured Urdu text file, transliterates it to Roman Urdu,
    translates it to English, and saves it in a structured format.
    """
    input_txt_file = "iqbal.txt"
    output_txt_file = "iqbal_knowledge_base.txt"

    print(f"Reading unstructured data from '{input_txt_file}'...")

    try:
        with open(input_txt_file, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_txt_file}' was not found.")
        return

    stanzas = content.strip().split('\n\n')
    translator = GoogleTranslator(source='ur', target='en')

    print(f"Found {len(stanzas)} stanzas. Processing, transliterating, and translating...")

    with open(output_txt_file, "w", encoding="utf-8") as f_out:
        for i, stanza in enumerate(stanzas):
            clean_stanza = "\n".join(line.strip() for line in stanza.split('\n') if line.strip())
            if not clean_stanza:
                continue

            try:
                # Use our built-in function
                roman_urdu = built_in_transliterate(clean_stanza)
                english_translation = translator.translate(clean_stanza)
                title = roman_urdu.split('\n')[0]

                f_out.write(f"(From: {title})\n")
                f_out.write(f"{roman_urdu}\n")
                f_out.write(f"English Translation: {english_translation}\n")
                f_out.write("---\n\n")

                print(f"Processed stanza {i+1}/{len(stanzas)}")
                time.sleep(0.2)

            except Exception as e:
                print(f"Could not process stanza {i+1}. Error: {e}")
                continue

    print("\nAutomated processing complete!")
    print(f"A structured knowledge base has been saved to '{output_txt_file}'.")

if __name__ == "__main__":
    auto_process_urdu_text()
