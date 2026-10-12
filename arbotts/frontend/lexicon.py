"""Indonesian lexical resolver for e-pepet (/ə/) and e-taling (/e/, /ɛ/).

Solves common espeak-ng mispronunciations in Indonesian phonology:
1. KBBI phonology rule: Open-ended word-final 'e' (sate, tempe, sore, kafe, jahe, etc.)
   is ALWAYS e-taling (/e/), never e-pepet (/ə/).
2. Common native roots where penultimate 'e' is e-pepet (/ə/) (beras, kera, tepi,
   teman, lezat, dekat, etc.), which espeak-ng wrongly predicts as stressed open /ɛ/.
3. Common roots where 'e' is e-taling (/e/, /ɛ/) (bebek, meja, capek, merah, desa, dll).
"""

from __future__ import annotations

import re

# Words ending in open 'e' are always taling in Indonesian phonology
# (KBBI: vokal e pada suku kata terbuka di akhir kata selalu dieja taling /e/)
OPEN_E_EXCEPTIONS = frozenset([
    "sate", "tempe", "sore", "kafe", "jahe", "lele", "tante", "kue", "kode",
    "mode", "ide", "tape", "toge", "onde", "bemo", "sate-sate", "tempe-tempe"
])

# Native root words and common derivations whose 'e' is pepet /ə/
# espeak-ng defaults to stressed open /ɛ/ on these words
PEPET_ROOTS = frozenset([
    # High frequency function and common words
    "dengan", "penuh", "beli", "membeli", "dibeli", "pembeli", "pembelian",
    "beri", "memberi", "diberi", "pemberian", "temu", "bertemu", "menemui",
    "kenal", "mengenal", "dikenal", "terkenal", "pegang", "memegang", "dipegang",
    "tekan", "menekan", "ditekan", "tekanan", "renang", "berenang", "perenang",
    "senin", "selasa", "kemarin", "sebentar", "sekarang", "selamat", "semoga",
    "selalu", "semua", "seperti", "sedang", "sendiri", "sembilan", "sepuluh",
    "sebelas", "seratus", "seribu", "sedikit", "tersebut", "pemerintah", "negeri",

    # Core vocabulary
    "beras", "kera", "tepi", "teman", "lezat", "dekat", "besar", "cepat",
    "tepat", "berat", "lebat", "tebal", "gelap", "terang", "tenang", "senang",
    "selat", "tepuk", "remuk", "belut", "semut", "leher", "lemah", "kering",
    "keluar", "kembali", "kertas", "kepala", "kelapa", "keluarga", "belum",
    "benar", "benci", "benda", "bengkok", "bentak", "bentang", "bentuk",
    "berani", "berapa", "berkas", "berkat", "bersih", "betul", "cegah",
    "cekik", "celaka", "celana", "cemburu", "cemas", "cengkeram", "cerah",
    "cerita", "cermat", "cermin", "derajat", "deras", "derita", "dermaga",
    "gedung", "geli", "gempa", "gemetar", "gerak", "geram", "gerbang", "gereja",
    "gertak", "hela", "hembus", "hening", "hentak", "henti", "jelajah",
    "jelas", "jelita", "jemput", "jemur", "jendela", "jenderal", "jengkal",
    "jengkel", "jerapah", "jerat", "jerit", "jeruk", "kelabu", "kelasi",
    "kelelawar", "keledai", "kelinci", "kelompok", "keluh", "kemari", "kemarau",
    "kemas", "kembang", "kembar", "kempa", "kempis", "kemudi", "kendali",
    "kendaraan", "kendur", "kental", "kentang", "kentut", "kenyal", "kenyang",
    "kepada", "kepal", "keping", "kepiting", "kepul", "kerak", "keramas",
    "keranjang", "kerbau", "kerikil", "keringat", "keripik", "keris", "keriting",
    "kerudung", "kerumun", "kesal", "kesan", "kesatria", "ketam", "ketat",
    "ketemu", "ketik", "ketika", "ketua", "ketuk", "ketupat", "lebah", "lebam",
    "lebar", "lebih", "lecut", "ledak", "lega", "legam", "lekat", "lekuk", "lelah",
    "lelaki", "leluasa", "lemak", "lemari", "lembab", "lembar", "lembut",
    "lempeng", "lencana", "lengah", "lengan", "lengkap", "lengkung", "lenyap",
    "lepas", "lereng", "lesu", "letak", "letup", "letus", "medan", "megah",
    "melati", "melarat", "menara", "mendarat", "menit", "mentah", "menteri",
    "merak", "merang", "merpati", "mesin", "meski", "mesra", "mewah",
    "neraka", "pedang", "pedas", "pedati", "peduli", "pegawai", "pekan",
    "pelan", "pelangi", "pelat", "pelatuk", "pelopor", "peluit", "peluk", "peluru",
    "pematang", "pemeo", "penat", "pendek", "pendekar", "pendeta", "pengantin",
    "penjara", "perahu", "perak", "perang", "perangkap", "perban", "perca",
    "percaya", "percik", "perdana", "perempuan", "pergi", "perih", "periksa",
    "perintah", "peristiwa", "perkara", "perkasa", "perlahan", "perlu",
    "permaisuri", "pernah", "perosot", "persen", "persis", "pertama", "pertiwi",
    "perut", "pesan", "pesantren", "pesat", "pesona", "petak", "petang", "petani",
    "peti", "petik", "petir", "petualang", "petunjuk", "rebah", "rebung", "rebus",
    "reda", "redam", "redup", "regan", "regang", "regu", "rehat", "rejam", "rekat",
    "rela", "relung", "remang", "remas", "remeh", "rempah", "remuk", "rencana",
    "rendah", "rendam", "rendang", "renggut", "renik", "rentan", "rentang",
    "renten", "renung", "repot", "resah", "resap", "resmi", "rezeki", "sebab",
    "sebal", "sebat", "sebelah", "seberang", "sebut", "sedap", "sedih", "sedot",
    "seduh", "segala", "segan", "segar", "segel", "segera", "segi", "seharga",
    "sehat", "sejak", "sejarah", "sejati", "sejengkal", "sejuk", "sekali",
    "sekat", "sekedar", "sekeliling", "sekejap", "sekolah", "sekoteng", "sekrup",
    "sela", "selada", "selam", "selang", "selaput", "selatan", "selempang",
    "selendang", "selenggara", "selesai", "selidik", "selimut", "selip",
    "selisih", "selokan", "selongsong", "seloroh", "seluler", "selundup", "seluruh",
    "semak", "semalam", "semangat", "semangka", "semarak", "semat", "sembah",
    "sembelih", "sembelit", "semberani", "semboyan", "sembuh", "sembunyi",
    "sembur", "semen", "semena", "semenanjung", "sementara", "semerbak", "semesta",
    "semi", "semir", "semprot", "sempurna", "semrawut", "senam", "senapan",
    "senat", "senda", "sendang", "sendu", "sengaja", "sengat", "senggol",
    "sengit", "sengketa", "seni", "seniman", "senja", "senjata", "sentak",
    "sentil", "sentimen", "sentosa", "sentra", "sentuh", "senyap", "senyum",
    "sepak", "sepakat", "sepanjang", "sepantun", "separuh", "sepat", "sepatu",
    "sepi", "sepoi", "seragam", "serah", "serai", "serak", "seram", "serambi",
    "serang", "serap", "serat", "serba", "serbak", "serban", "serbuan", "serbuk",
    "serdadu", "serentak", "sergap", "seri", "serikat", "sering", "serius",
    "serpih", "serta", "seru", "seruling", "sesak", "sesal", "sesap", "sesar",
    "sesat", "setan", "setara", "setia", "setiap", "setir", "setrika", "setrum",
    "setuju", "tebak", "tebang", "tebas", "tebat", "tebing", "tebu", "tebus",
    "teduh", "tegah", "tegar", "tegun", "tegur", "teka", "tekad", "tekat",
    "tekik", "teko", "tekuk", "tekur", "telaah", "teladan", "telaga", "telah",
    "telanjang", "telanjur", "telantar", "telap", "telapak", "telaten", "teliti",
    "telor", "teluk", "telungkup", "telunjuk", "telur", "tembaga", "tembak",
    "tembakau", "tembam", "tembelang", "tembereh", "tembereng", "tembesu",
    "tembiang", "tembikar", "tembilang", "tembuni", "tembus", "tempa", "tempat",
    "tempawan", "tempayan", "tempeh", "tempel", "temperamen", "tempias",
    "tempik", "tempo", "tempuh", "tempur", "tempurung", "tenaga", "tengadah",
    "tengger", "tengik", "tengkar", "tengking", "tengkorak", "tengku",
    "tengkulak", "tengkuk", "tengkurap", "tengok", "tengsin", "tengu", "tenis",
    "tentamen", "tentang", "tentara", "tenteng", "tenteram", "tentu", "tenun",
    "tenung", "tepak", "tepam", "tepas", "tepaut", "tepekur", "tepis", "tepung",
    "teraju", "terak", "teral", "terali", "teram", "teran", "terap", "terapung",
    "teras", "terasi", "teratai", "terawang", "terba", "terbis", "terbus", "teri",
    "teriak", "terik", "terka", "terkam", "terkapar", "terkup", "terlak",
    "terlanjur", "terlentang", "terma", "terman", "termangu", "termas", "terpa",
    "terpas", "terpaan", "terpekur", "terpencil", "terpelanting", "terpeluk",
    "terpening", "terperanjat", "terperosok", "terpesona", "terpincut", "terpuk",
    "terputus", "tersangka", "tersaput", "terselip", "tersengal", "tersenyum",
    "terserah", "tersesat", "tersiksa", "tersinggung", "tersungkur", "tertata",
    "terung", "teruntum", "terup", "terus", "tetak", "tetal", "tetampah",
    "tetamu", "tetangga", "tetap", "tetapi", "tetas", "tetes", "tetiba", "tetirah"
])

PEPET_PREFIXES = (
    "meng", "meny", "peng", "peny", "mem", "men", "pem", "pen",
    "ber", "ter", "per", "bel", "pel", "me", "pe", "se", "ke", "di"
)

CLITICS = ("-nya", "-ku", "-mu", "-lah", "-kah", "-pun", "-tah")


def _get_candidates(word: str) -> list[str]:
    """Extract root and base candidates considering clitics and affixes."""
    w = word.lower().strip()
    w = re.sub(r"^[^\w]+|[^\w]+$", "", w)
    if not w:
        return []

    candidates: list[str] = [w]

    # Strip hyphenated clitics like senin-mu, temannya
    for cl in CLITICS:
        if w.endswith(cl):
            base_w = w[:-len(cl)]
            candidates.append(base_w)
            w = base_w
            break

    # Strip reduplication hyphen like kera-kera, teman-teman
    if "-" in w:
        parts = w.split("-")
        candidates.extend(parts)
        w = parts[0]

    # Morphological prefix stripping and nasalization reversal
    for pref in PEPET_PREFIXES:
        if w.startswith(pref) and len(w) > len(pref) + 2:
            rem = w[len(pref):]
            candidates.append(rem)
            if pref in ("meng", "peng"):
                candidates.append("k" + rem)
            elif pref in ("mem", "pem"):
                candidates.append("p" + rem)
            elif pref in ("men", "pen"):
                candidates.append("t" + rem)
            elif pref in ("meny", "peny"):
                candidates.append("s" + rem)
            elif pref == "ber" and rem.startswith("e"):
                candidates.append("r" + rem[1:])
            break

    return candidates


def resolve_indonesian_phonemes(word: str, phoneme: str) -> str:
    """Adjust espeak-ng phoneme for an Indonesian word using lexical rules."""
    w_clean = re.sub(r"^[^\w]+|[^\w]+$", "", word).lower()
    p = phoneme

    # Rule 1: Words ending in open 'e' (sate, tempe, sore, kafe, etc.)
    # In Indonesian phonology, open word-final 'e' is always taling /e/, never schwa /ə/.
    if (w_clean != "ke" and w_clean.endswith("e") and len(w_clean) >= 3) or w_clean in OPEN_E_EXCEPTIONS:
        p = re.sub(r"ə([.,!?;:\'\"\)\]]*)$", r"e\1", p)

    # Rule 2: Root or derived word is a known pepet word
    # espeak-ng mispredicts penultimate pepet as stressed open /ɛ/.
    candidates = _get_candidates(word)
    if any(c in PEPET_ROOTS for c in candidates):
        p = p.replace("ɛ", "ə")

    return p
