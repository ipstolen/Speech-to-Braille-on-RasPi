"""Braille translation module using pybrl."""

import pybrl
from typing import Optional

from ..config import BRAILLE_TABLE, BRAILLE_LANGUAGE


class BrailleTranslator:
    """Handles translation of text to Braille using pybrl."""

    def __init__(self, table: str = BRAILLE_TABLE, language: str = BRAILLE_LANGUAGE):
        self.table = table
        self.language = language

    def translate(self, text: str) -> Optional[str]:
        """
        Translate text to Braille Unicode symbols.

        Args:
            text: English text to translate

        Returns:
            Braille translation as Unicode symbols, or None if translation fails
        """
        if not text or not text.strip():
            return None

        try:
            # Clean text - pybrl may have issues with certain punctuation
            # Remove problematic characters but keep basic punctuation
            cleaned_text = text.replace('"', '').replace('"', '').replace('"', '')

            # Use pybrl to translate to Braille
            braille_output = pybrl.translate(cleaned_text, main_language=self.language)

            if braille_output:
                # Convert to Unicode symbols
                unicode_braille = pybrl.toUnicodeSymbols(braille_output, flatten=True)
                return unicode_braille
            else:
                return None

        except Exception as e:
            print(f"Braille translation error: {e}")
            print(f"Failed text: '{text}'")
            # Try again with more aggressive cleaning
            try:
                # Keep only alphanumeric, spaces, and basic punctuation
                import re
                safe_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'-]', '', text)
                braille_output = pybrl.translate(safe_text, main_language=self.language)
                if braille_output:
                    unicode_braille = pybrl.toUnicodeSymbols(braille_output, flatten=True)
                    return unicode_braille
            except:
                pass
            return None

    def to_unicode(self, text: str) -> str:
        """
        Translate text to Braille Unicode symbols.

        This is an alias for translate().

        Args:
            text: English text to translate

        Returns:
            Braille translation as Unicode symbols
        """
        result = self.translate(text)
        return result if result else ""

    def translate_raw(self, text: str) -> Optional[any]:
        """
        Translate text to raw Braille representation (before Unicode conversion).

        Args:
            text: English text to translate

        Returns:
            Raw Braille output from pybrl, or None if translation fails
        """
        if not text or not text.strip():
            return None

        try:
            braille_output = pybrl.translate(text, main_language=self.language)
            return braille_output

        except Exception as e:
            print(f"Braille translation error: {e}")
            return None
