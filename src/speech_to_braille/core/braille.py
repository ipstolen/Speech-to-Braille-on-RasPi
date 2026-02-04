"""Braille translation module."""

# import pybrl
from typing import Optional

from ..config import BRAILLE_TABLE, BRAILLE_LANGUAGE


class BrailleTranslator:
    """Handles translation of text to Braille."""

    def __init__(self, table: str = BRAILLE_TABLE, language: str = BRAILLE_LANGUAGE):
        self.table = table
        self.language = language
        # self.translator = pybrl.load_table(table)

    def translate(self, text: str) -> Optional[str]:
        """Translate text to Braille"""
        # Braille translation currently not implemented
        # Uncomment below when pybrl module is properly integrated
        # output_braille = pybrl.translate(text, main_language=self.language)
        # return pybrl.toUnicodeSymbols(output_braille, flatten=True)
        return None

    def to_unicode(self, braille_text: str) -> str:
        """Convert Braille to Unicode symbols"""
        # output = pybrl.toUnicodeSymbols(braille_text, flatten=True)
        # return output
        return ""
