"""
KhmerXScore Preprocessing Pipeline
===================================
1. Unicode NFC normalization
2. Remove zero-width characters (ZWSP, ZWNJ, FEFF)
3. KCC character cluster canonical ordering
4. Clean \\n literal strings from CSV
5. Normalize whitespace
6. Optional: khmernltk word segmentation (for ML/DL tiers + PrahokBART)
NO spellcheck — preserves student mistakes that teachers scored
"""
import re
import string
import unicodedata

# ASCII punctuation + Khmer punctuation (U+17D4–U+17DA: khan, bariyoosan,
# camnuc pii kuuh, lek todo, beyyal, phnaek muan, koomuut)
_PUNCT_RE = re.compile(r"[" + re.escape(string.punctuation) + r"។-៚]")


class KhmerPreprocessor:
    def __init__(self, segment=False):
        self.segment = segment
        self._seg_fn = None
        if segment:
            try:
                import khmernltk
                self._seg_fn = khmernltk.word_tokenize
            except ImportError:
                print("WARNING: khmernltk not installed. pip install khmernltk")
                print("Using space-based fallback")

    def __call__(self, text):
        if not text or not isinstance(text, str):
            return ""
        text = str(text).strip()
        if not text:
            return ""

        # 1. Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # 2. Remove zero-width characters
        text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

        # 3. Remove control characters
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # 4. KCC canonical ordering
        text = self._kcc_normalize(text)

        # 5. Strip punctuation (ASCII + Khmer sentence marks)
        text = _PUNCT_RE.sub(" ", text)

        # 6. Clean literal \\n from CSV
        text = text.replace("\\n", " ").replace("\n", " ")

        # 7. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 7. Optional segmentation
        if self.segment and self._seg_fn:
            try:
                tokens = self._seg_fn(text)
                text = " ".join(tokens)
            except Exception:
                pass

        return text

    def _kcc_normalize(self, text):
        """Canonical ordering within each Khmer syllable cluster.

        A new cluster starts at each base consonant not preceded by coeng (U+17D2).
        Coeng + consonant is a subscript and stays inside the current syllable.
        """
        result = []
        cluster = []
        prev_was_coeng = False
        for ch in text:
            cp = ord(ch)
            is_khmer = 0x1780 <= cp <= 0x17FF or 0x19E0 <= cp <= 0x19FF
            is_base = 0x1780 <= cp <= 0x17A2
            is_coeng = cp == 0x17D2
            if is_khmer:
                if is_base and not prev_was_coeng and cluster:
                    result.append(self._sort_cluster(cluster))
                    cluster = []
                cluster.append(ch)
                prev_was_coeng = is_coeng
            else:
                if cluster:
                    result.append(self._sort_cluster(cluster))
                    cluster = []
                result.append(ch)
                prev_was_coeng = False
        if cluster:
            result.append(self._sort_cluster(cluster))
        return "".join(result)

    def _sort_cluster(self, chars):
        """Sort: base consonant → coeng+subscript (sorted) → vowels → signs"""
        base, subs, vowels, signs, other = [], [], [], [], []
        i, found_base = 0, False
        while i < len(chars):
            cp = ord(chars[i])
            if 0x1780 <= cp <= 0x17A2 and not found_base:
                base.append(chars[i])
                found_base = True
            elif cp == 0x17D2 and i + 1 < len(chars) and 0x1780 <= ord(chars[i + 1]) <= 0x17A2:
                subs.append((chars[i], chars[i + 1]))
                i += 1
            elif 0x17B6 <= cp <= 0x17C5:
                vowels.append(chars[i])
            elif 0x17C6 <= cp <= 0x17D1:
                signs.append(chars[i])
            else:
                other.append(chars[i])
            i += 1
        r = base[:]
        for co, cn in sorted(subs, key=lambda x: ord(x[1])):
            r.extend([co, cn])
        r.extend(vowels)
        r.extend(signs)
        r.extend(other)
        return "".join(r)
