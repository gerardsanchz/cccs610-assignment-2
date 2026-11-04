"""CCCS 610, session 7: word clouds for Louis Hemon, Maria Chapdelaine.

Downloads Project Gutenberg ebook 13525 once (data/maria-chapdelaine-13525.txt),
cuts the novel out of the transcriber's front matter, then runs two pipelines:

* the class pipeline of the session 7 notebook, unchanged except that the book
  is French (English stop words are kept on purpose, to measure what breaks);
* an improved pipeline (accents kept, French stop words, plural and feminine
  merging, character names kept or removed, bigrams, per-chapter lift).

Writes figures/m07-cloud-class.png, figures/m07-cloud-fixed.png,
figures/m07-cloud-no-names.png and results/m07.json next to this folder.

Run (any directory):
    python -I code/m07_words.py
"""

import json
import re
import string
import unicodedata
from collections import Counter
from itertools import pairwise
from math import log
from pathlib import Path
from typing import Any

from nltk.corpus import stopwords
from PIL import ImageFont
from wordcloud import WordCloud

SEED = 42
BOOK_ID = 13525
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / f"maria-chapdelaine-{BOOK_ID}.txt"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results" / "m07.json"

# The 15 names of the brief (token form, lowercase); Tit'Bé loses its
# apostrophe before tokenizing so it survives the split on apostrophes.
NAMES = [
    "maria", "samuel", "chapdelaine", "laura", "eutrope", "gagnon", "françois",
    "paradis", "lorenzo", "surprenant", "télesphore", "titbé", "éphrem",
    "chaillon", "légaré",
]  # fmt: skip
# Compound names written with an apostrophe or hyphen, joined before tokenizing.
PROTECTED = re.compile(r"\b(Tit|Da)['’]([^\W\d_]+)|\b(Alma)-(Rose)")
# Other characters that dominate the cloud once the 15 names are gone;
# "chaillon" of the brief never occurs in this edition (count 0).
NAMES_EXTRA = [
    "esdras", "dabé", "almarose", "azalma", "edwige", "edwidge", "wilfrid", "titsèbe",
]  # fmt: skip
# Words that the plural/feminine rule would fuse with a different word
# (marie/mari, maine/main, coupe/coup, vente/vent, ...), found by reading the
# merge groups with a stem of four letters or fewer.
NO_MERGE = [
    "marie", "maine", "coupe", "vente", "base", "cuire", "fonte", "lune",
    "lunes", "dise", "vite", "prise", "mise", "sorte", "sortes", "cours",
    "fils", "rase", "rites", "rite", "crie", "close", "closes",
]  # fmt: skip

# Elided clitics left behind when the apostrophe becomes a space (l'eglise).
ELIDED = ["l", "d", "qu", "j", "n", "s", "c", "m", "t",
          "jusqu", "lorsqu", "puisqu", "quoiqu", "presqu", "quelqu"]  # fmt: skip
# Function words still dominating after the nltk list (see residual_top in the
# JSON): adverbs, pronouns, quantifiers and the dialogue verbs dit/fit, which
# tag every line of speech and say nothing about the book.
ADDED_STOP = [
    "plus", "comme", "tout", "tous", "toute", "toutes", "bien", "sans",
    "encore", "quand", "peu", "dit", "fit", "faire", "fait", "dire", "rien",
    "après", "près", "avant", "vers", "autre", "autres", "quelque",
    "quelques", "aussi", "leurs", "puis", "être", "peut", "toujours", "cela",
    "cette", "presque", "déjà", "fois", "deux", "trois", "chaque", "chacun",
    "alors", "ainsi", "dont", "avoir", "faut", "vint",
]  # fmt: skip

ACCENTED = "éèêàçôûâîùëïÉÀÈÊÔÇ"


def load_book() -> str:
    """Download once through gutenbergpy, cache as UTF-8 text."""
    if not DATA.exists():
        import gutenbergpy.textget

        raw = gutenbergpy.textget.get_text_by_id(BOOK_ID)
        content = gutenbergpy.textget.strip_headers(raw)
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(content.decode("utf-8"), encoding="utf-8", newline="\n")
    return unicodedata.normalize("NFC", DATA.read_text(encoding="utf-8"))


def cut_novel(text: str) -> tuple[str, dict[str, Any]]:
    """Cut from the heading CHAPITRE I to the last sentence of the ebook."""
    start_match = re.search(r"^CHAPITRE I\s*$", text, flags=re.MULTILINE)
    if start_match is None:
        raise ValueError("start marker 'CHAPITRE I' not found")
    end_marker = "quand les hommes reviendront du\nbois pour les semailles."
    end = text.rindex(end_marker) + len(end_marker)
    book = text[start_match.start() : end]
    info = {
        "start_marker": "first line equal to 'CHAPITRE I' (regex ^CHAPITRE I$)",
        "end_marker": "last occurrence of 'bois pour les semailles.' "
        "(final sentence; the ebook ends there)",
        "start_index": start_match.start(),
        "end_index": end,
        "stripped_text_chars": len(text),
        "novel_chars": len(book),
        "head_removed": text[: start_match.start()].strip().splitlines()[:3],
        "first_200": book[:200],
        "last_200": book[-200:],
    }
    return book, info


def top(counter: Counter[str], n: int) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def make_cloud(
    text_or_freq: str | dict[str, float], out: Path, class_mode: bool
) -> WordCloud:
    """700x700 like the class; fixed random_state for identical output."""
    if class_mode:
        cloud = WordCloud(width=700, height=700, random_state=SEED)
        cloud.generate(str(text_or_freq))
    else:
        cloud = WordCloud(
            width=700, height=700, random_state=SEED, background_color="white",
            stopwords=set(), collocations=False,
        )  # fmt: skip
        assert isinstance(text_or_freq, dict)
        cloud.generate_from_frequencies(text_or_freq)
    out.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_file(str(out))
    return cloud


def non_ascii_words(cloud: WordCloud) -> list[str]:
    return [w for w in cloud.words_ if not w.isascii()]


# ---------------------------------------------------------------- class pipeline
def class_pipeline(book: str) -> dict[str, Any]:
    """Cells of the session 7 notebook, in order, English stop words kept."""
    punct = string.punctuation
    removed = book.translate(str.maketrans(punct, " " * len(punct)))
    forbid = f"[^{re.escape(string.printable)}]"
    ok = re.sub(forbid, " ", removed)
    cleaner = re.sub(r"\s+", " ", ok)
    words = cleaner.split()

    good = [w for w in words if len(w) < 2 or not w.isupper()]
    sw = list(stopwords.words("english")) + ["mr", "mrs"]
    sw_set = set(sw)
    better = [
        w for w in good
        if w.lower() not in sw_set and not w.isdigit() and len(w) > 2
    ]  # fmt: skip

    distinct = set(better)

    def singularize(word: str) -> str:
        lc = word.lower()
        if lc in distinct:
            word = lc
        if word[-1] == "s":
            cropped = word[:-1]
            if cropped in distinct:
                return cropped
        return word

    simple = [singularize(w) for w in better]
    freq = Counter(simple)

    # Damage by the printable filter, measured per whitespace token of `removed`.
    # Tag every surviving piece with whether it came from a damaged token.
    pieces_tagged: list[tuple[str, bool]] = []
    n_forbidden_tokens = n_letter_damaged = n_split = n_one_piece = n_vanished = 0
    lost: Counter[str] = Counter()
    for tok in removed.split():
        bad = re.findall(forbid, tok)
        pieces = [p for p in re.split(forbid, tok) if p]
        damaged = any(ch.isalpha() for ch in bad)
        if bad:
            n_forbidden_tokens += 1
            lost.update(bad)
        if damaged:
            n_letter_damaged += 1
            if len(pieces) >= 2:
                n_split += 1
            elif len(pieces) == 1:
                n_one_piece += 1
            else:
                n_vanished += 1
        pieces_tagged.extend((p, damaged) for p in pieces)
    assert [p for p, _ in pieces_tagged] == words

    # Provenance per surviving token, followed through the same filters.
    frag_share: dict[str, float] = {}
    tagged_kept = [
        (p, d) for p, d in pieces_tagged
        if (len(p) < 2 or not p.isupper())
        and p.lower() not in sw_set and not p.isdigit() and len(p) > 2
    ]  # fmt: skip
    assert len(tagged_kept) == len(better)
    from_damaged: Counter[str] = Counter()
    for (_, d), s in zip(tagged_kept, simple):
        if d:
            from_damaged[s] += 1
    for w, _ in top(freq, 100):
        frag_share[w] = from_damaged[w] / freq[w]

    top25 = [
        {"word": w, "count": c, "fragment_share": round(frag_share[w], 3)}
        for w, c in top(freq, 25)
    ]
    fragments = [t["word"] for t in top25 if t["fragment_share"] >= 0.5]
    fragments_top100 = [
        w for w, _ in top(freq, 100) if frag_share[w] >= 0.5
    ]  # fragment_share: part of a word's occurrences cut out of an accented word

    dropped_sw = sum(1 for w in good if w.lower() in sw_set)
    dropped_digit = sum(1 for w in good if w.lower() not in sw_set and w.isdigit())
    dropped_short = sum(
        1 for w in good
        if w.lower() not in sw_set and not w.isdigit() and len(w) <= 2
    )  # fmt: skip

    steps = {
        "book_whitespace_tokens": len(book.split()),
        "after_punctuation_to_space": len(removed.split()),
        "after_printable_filter": len(ok.split()),
        "after_whitespace_collapse": len(words),
        "after_drop_allcaps": len(good),
        "after_stopwords_digits_len3": len(better),
        "after_plural_merge": len(simple),
    }
    return {
        "steps_tokens": steps,
        "distinct_before_merge": len(distinct),
        "distinct_after_merge": len(set(simple)),
        "allcaps_dropped": len(words) - len(good),
        "dropped_by_english_stopwords": dropped_sw,
        "dropped_as_digits": dropped_digit,
        "dropped_as_two_chars_or_fewer": dropped_short,
        "top100_fragments_of_accented_words": fragments_top100,
        "chars_removed_by_printable_filter": sum(lost.values()),
        "removed_chars_top": lost.most_common(12),
        "tokens_with_any_removed_char": n_forbidden_tokens,
        "tokens_with_removed_letter": n_letter_damaged,
        "tokens_split_in_two_or_more": n_split,
        "tokens_left_as_one_truncated_piece": n_one_piece,
        "tokens_vanished_entirely": n_vanished,
        "top25": top25,
        "top25_fragments_of_accented_words": fragments,
        "_simple": simple,
        "_distinct_words": len(distinct),
    }


# -------------------------------------------------------------- improved pipeline
LETTERS = re.compile(r"[^\W\d_]+")


def protect(text: str) -> str:
    """Join Tit'Bé, Da'Bé and Alma-Rose so the apostrophe split keeps them whole."""
    return PROTECTED.sub(lambda m: "".join(g for g in m.groups() if g), text)


def tokenize(text: str) -> tuple[list[str], dict[str, int]]:
    """Unicode letters only; apostrophes and hyphens split; ALLCAPS dropped."""
    text = protect(text)
    raw = LETTERS.findall(text)
    n_digits = len(re.findall(r"\d+", text))
    kept = [w for w in raw if len(w) < 2 or not w.isupper()]
    return [w.lower() for w in kept], {
        "letter_tokens": len(raw),
        "allcaps_dropped": len(raw) - len(kept),
        "digit_tokens_never_tokenized": n_digits,
    }


def shape(tokens: list[str]) -> dict[str, Any]:
    c = Counter(tokens)
    return {
        "tokens": len(tokens),
        "distinct": len(c),
        "tokens_with_accent": sum(1 for t in tokens if not t.isascii()),
        "top10": top(c, 10),
    }


def build_merge_map(vocab: set[str]) -> dict[str, str]:
    """The class rule (trailing s) and a feminine rule (trailing e)."""
    protected = set(NAMES) | set(NAMES_EXTRA) | set(NO_MERGE)
    mapping: dict[str, str] = {}
    for w in sorted(vocab):
        out = w
        if w not in protected:
            if out.endswith("s") and out[:-1] in vocab:
                out = out[:-1]
            if out.endswith("e") and out[:-1] in vocab:
                out = out[:-1]
        mapping[w] = out
    return mapping


def improved_pipeline(book: str) -> dict[str, Any]:
    tokens, tok_info = tokenize(book)
    en = set(stopwords.words("english")) | {"mr", "mrs"}
    fr = set(stopwords.words("french"))
    added = set(ADDED_STOP)
    elided = Counter(t for t in tokens if t in ELIDED)

    def keep_len(ts: list[str]) -> list[str]:
        return [t for t in ts if t not in ELIDED and len(t) > 2]

    stages: dict[str, dict[str, Any]] = {}
    s0 = tokens
    stages["a_unicode_tokens_lowercase_only"] = shape(s0)
    s1 = [t for t in keep_len(s0) if t not in en]
    stages["a_plus_len3_elision_and_english_stopwords"] = shape(s1)
    s2 = [t for t in keep_len(s0) if t not in fr]
    stages["b1_french_nltk_stopwords_instead"] = shape(s2)
    residual = Counter(s2)
    s3 = [t for t in s2 if t not in added]
    stages["b2_plus_added_function_words"] = shape(s3)

    vocab = set(s3)
    mapping = build_merge_map(vocab)
    s4 = [mapping[t] for t in s3]
    keep_whole = set(NAMES) | set(NAMES_EXTRA) | set(NO_MERGE)
    plural_only = {
        w: w[:-1] if w.endswith("s") and w[:-1] in vocab and w not in keep_whole
        else w
        for w in vocab
    }  # fmt: skip
    stages["c1_plural_merge_only"] = shape([plural_only[t] for t in s3])
    stages["c2_plural_and_feminine_merge"] = shape(s4)

    groups: dict[str, set[str]] = {}
    for w, canon in mapping.items():
        groups.setdefault(canon, set()).add(w)
    freq3 = Counter(s3)
    merges = sorted(
        ((canon, sorted(m, key=lambda x: (-freq3[x], x)))
         for canon, m in groups.items() if len(m) > 1),
        key=lambda kv: (-sum(freq3[x] for x in kv[1]), kv[0]),
    )  # fmt: skip
    short_stem_merges = [[canon, m] for canon, m in merges if len(canon) <= 4]

    freq = Counter(s4)
    names_all = set(NAMES)
    extras = set(NAMES_EXTRA)
    no_names_tokens = [t for t in s4 if t not in names_all and t not in extras]
    no_names_freq = Counter(no_names_tokens)
    stages["d_names_removed"] = shape(no_names_tokens)

    return {
        "tokenization": tok_info,
        "elided_forms_dropped": dict(sorted(elided.items())),
        "elided_tokens_dropped_total": sum(elided.values()),
        "len_lt3_non_elided_dropped": sum(
            1 for t in s0 if t not in ELIDED and len(t) <= 2
        ),
        "english_stopwords_removed_from_french": sum(
            1 for t in keep_len(s0) if t in en
        ),
        "english_stopwords_top": top(Counter(t for t in keep_len(s0) if t in en), 12),
        "french_nltk_stopword_tokens_removed": sum(1 for t in keep_len(s0) if t in fr),
        "added_function_words": {
            w: residual[w] for w in sorted(added, key=lambda x: (-residual[x], x))
        },
        "added_function_words_tokens_removed": sum(residual[w] for w in added),
        "residual_top_after_nltk": top(residual, 70),
        "stages": stages,
        "merge_groups_top": [[c, m] for c, m in merges[:25]],
        "merge_groups_total": len(merges),
        "merge_groups_short_stem": short_stem_merges,
        "top25": top(freq, 25),
        "top25_no_names": top(no_names_freq, 25),
        "_s3": s3,
        "_mapping": mapping,
        "_freq": freq,
        "_no_names_freq": no_names_freq,
    }


def name_counts(book: str, tokens_total: int) -> dict[str, Any]:
    shown = {
        "titbé": "Tit'Bé",
        "dabé": "Da'Bé",
        "almarose": "Alma-Rose",
        "titsèbe": "Tit'Sèbe",
    }
    out: dict[str, Any] = {}
    for name in NAMES + NAMES_EXTRA:
        form = shown.get(name, name.capitalize())
        edge = r"(?<![^\W\d_]){}(?![^\W\d_])"
        n_any = len(re.findall(edge.format(re.escape(form)), book, flags=re.IGNORECASE))
        n_cap = len(re.findall(edge.format(re.escape(form)), book))
        out[name] = {"form": form, "any_case": n_any, "capitalized_form": n_cap}
    out["_note"] = (
        "any_case = occurrences ignoring case; capitalized_form = occurrences "
        "written exactly as in 'form' (a common noun such as paradis or rose "
        f"also occurs in lowercase); {tokens_total} letter tokens in the book"
    )
    return out


def bigrams(book: str, keep_map: dict[str, str]) -> list[tuple[str, int]]:
    """Adjacent pairs inside one sentence where both words survive the filters.

    keep_map maps every surviving raw word to its merged form; adjacency is
    taken in the original token sequence so a removed stop word breaks the pair.
    """
    counts: Counter[str] = Counter()
    text = protect(book)
    for seg in re.split(r"[.!?…;:«»]+|\n\s*\n|--", text):
        toks = [t.lower() for t in LETTERS.findall(seg) if not t.isupper()]
        for a, b in pairwise(toks):
            if a in keep_map and b in keep_map:
                counts[f"{keep_map[a]} {keep_map[b]}"] += 1
    return top(counts, 15)


def chapters(book: str) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    heads = list(re.finditer(r"^CHAPITRE ([IVX]+)\s*$", book, flags=re.MULTILINE))
    spans = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(book)
        spans.append((m.group(1), book[m.end() : end]))
    info = {
        "method": "regex ^CHAPITRE ([IVX]+)$ over the cut novel, one chapter "
        "per heading to the next heading",
        "headings_found": len(heads),
        "numerals": [m.group(1) for m in heads],
        "note": "16 chapters in this edition, not 12",
    }
    return spans, info


def chapter_lift(
    spans: list[tuple[str, str]],
    mapping: dict[str, str],
    freq: Counter[str],
    min_count: int = 3,
) -> dict[str, Any]:
    """Share of a word in the chapter divided by its share in the whole book."""
    book_total = sum(freq.values())
    out: dict[str, Any] = {}
    for numeral, body in spans:
        toks, _ = tokenize(body)
        kept = [mapping[t] for t in toks if t in mapping]
        c = Counter(kept)
        n = len(kept)
        rows = []
        for w, k in c.items():
            if k >= min_count:
                lift = (k / n) / (freq[w] / book_total)
                rows.append((w, k, freq[w], round(lift, 2), round(k * log(lift), 2)))
        # Lift alone rewards a word seen three times and nowhere else; weighting
        # it by the chapter count (k * ln lift) keeps frequent, distinctive words.
        by_lift = sorted(rows, key=lambda r: (-r[3], -r[1], r[0]))[:5]
        by_keyness = sorted(rows, key=lambda r: (-r[4], r[0]))[:5]
        out[numeral] = {
            "tokens": n,
            "top5_by_lift": by_lift,
            "top5_by_count_weighted_lift": by_keyness,
        }
    return out


def font_check(cloud: WordCloud) -> dict[str, Any]:
    font = ImageFont.truetype(cloud.font_path, 40)
    notdef = bytes(font.getmask("\U0010ffff"))
    bad = [
        ch for ch in ACCENTED
        if bytes(font.getmask(ch)) == notdef or not any(bytes(font.getmask(ch)))
    ]  # fmt: skip
    return {
        "font": Path(cloud.font_path).name,
        "accented_glyphs_missing": bad,
        "ok": not bad,
    }


def main() -> None:
    text = load_book()
    book, cut = cut_novel(text)
    print("FIRST 200:", repr(cut["first_200"]))
    print("LAST 200:", repr(cut["last_200"]))

    cls = class_pipeline(book)
    cloud_c = make_cloud(
        " ".join(cls["_simple"]), FIGURES / "m07-cloud-class.png", True
    )

    imp = improved_pipeline(book)
    freq: Counter[str] = imp["_freq"]
    nn_freq: Counter[str] = imp["_no_names_freq"]
    cloud_f = make_cloud(dict(freq), FIGURES / "m07-cloud-fixed.png", False)
    cloud_n = make_cloud(dict(nn_freq), FIGURES / "m07-cloud-no-names.png", False)

    spans, ch_info = chapters(book)
    lift = chapter_lift(spans, imp["_mapping"], freq)
    keep_map = {w: imp["_mapping"][w] for w in imp["_mapping"]}
    bi = bigrams(book, keep_map)
    names = name_counts(book, imp["tokenization"]["letter_tokens"])
    name_tokens_removed = sum(freq[n] for n in NAMES)
    extra_tokens_removed = sum(freq[n] for n in NAMES_EXTRA)

    fonts = font_check(cloud_f)
    clouds = {
        "class": {"words": list(cloud_c.words_), "non_ascii": non_ascii_words(cloud_c)},
        "fixed": {"words": list(cloud_f.words_), "non_ascii": non_ascii_words(cloud_f)},
        "no_names": {
            "words": list(cloud_n.words_),
            "non_ascii": non_ascii_words(cloud_n),
        },
        "font": fonts,
        "wordcloud_params": "700x700, random_state=42; class: defaults "
        "(English STOPWORDS, collocations on); improved: frequencies, "
        "no stopwords, collocations off",
    }
    sizes = {p.name: p.stat().st_size for p in sorted(FIGURES.glob("m07-cloud-*.png"))}

    result = {
        "book": {"gutenberg_id": BOOK_ID, "title": "Maria Chapdelaine"},
        "cut": cut,
        "class_pipeline": {k: v for k, v in cls.items() if not k.startswith("_")},
        "improved_pipeline": {k: v for k, v in imp.items() if not k.startswith("_")},
        "names": names,
        "names_tokens_removed_brief_list": name_tokens_removed,
        "names_tokens_removed_extra_list": extra_tokens_removed,
        "bigrams_top15": bi,
        "chapters": ch_info,
        "chapter_distinctive_words": lift,
        "chapter_lift_rule": "(count in chapter / tokens in chapter) / "
        "(count in book / tokens in book); words with >= 3 occurrences in "
        "the chapter; columns word, count in chapter, count in book, lift, "
        "count * ln(lift)",
        "clouds": clouds,
        "png_bytes": sizes,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("class cloud words:", clouds["class"]["words"])
    print("fixed cloud non-ASCII words:", clouds["fixed"]["non_ascii"])
    print("no-names cloud non-ASCII words:", clouds["no_names"]["non_ascii"])
    print("font:", fonts)
    print("png bytes:", sizes)
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
