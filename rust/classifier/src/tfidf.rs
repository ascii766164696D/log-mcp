use crate::model::TfidfConfig;

/// Sparse feature vector: list of (feature_index, value) pairs.
pub type SparseVec = Vec<(usize, f64)>;

/// Check if a byte is a Unicode word character (for ASCII fast path).
/// Matches Python's `\w` for ASCII: [a-zA-Z0-9_]
#[inline(always)]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Extract words (2+ word chars) from text, matching `(?u)\b\w\w+\b`.
/// Returns byte ranges into the text.
/// For ASCII text, uses a fast byte scan. For non-ASCII, falls back to char iteration.
fn tokenize_words(text: &str) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut words = Vec::with_capacity(32);

    if text.is_ascii() {
        // Fast ASCII path: scan bytes directly
        let mut i = 0;
        while i < len {
            if is_word_byte(bytes[i]) {
                let start = i;
                i += 1;
                while i < len && is_word_byte(bytes[i]) {
                    i += 1;
                }
                if i - start >= 2 {
                    words.push((start, i));
                }
            } else {
                i += 1;
            }
        }
    } else {
        // Non-ASCII: iterate chars to handle Unicode word characters
        let mut start = None;
        let mut byte_start = 0;
        for (byte_idx, ch) in text.char_indices() {
            let is_word = ch.is_alphanumeric() || ch == '_';
            match (start, is_word) {
                (None, true) => {
                    start = Some(byte_idx);
                    byte_start = byte_idx;
                }
                (Some(_), false) => {
                    if byte_idx - byte_start >= 2 {
                        words.push((byte_start, byte_idx));
                    }
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(_) = start {
            if text.len() - byte_start >= 2 {
                words.push((byte_start, text.len()));
            }
        }
    }
    words
}

/// Thread-local buffers for word TF-IDF.
struct WordBuffers {
    ngram_buf: Vec<u8>,
    counts: Vec<u16>,
    dirty: Vec<usize>,
}

thread_local! {
    static WORD_BUFS: std::cell::RefCell<WordBuffers> = std::cell::RefCell::new(WordBuffers {
        ngram_buf: Vec::with_capacity(128),
        counts: Vec::new(),
        dirty: Vec::with_capacity(64),
    });
}

/// Word-level TF-IDF vectorization.
///
/// Matches sklearn's TfidfVectorizer with analyzer="word".
/// Tokenizes with `\b\w\w+\b` (manual scan), generates n-grams,
/// looks up vocab, computes TF*IDF, L2-normalizes.
pub fn vectorize_word(text: &str, config: &TfidfConfig) -> SparseVec {
    WORD_BUFS.with(|bufs| {
        let mut bufs = bufs.borrow_mut();
        vectorize_word_inner(text, config, &mut bufs)
    })
}

fn vectorize_word_inner(text: &str, config: &TfidfConfig, bufs: &mut WordBuffers) -> SparseVec {
    let word_ranges = tokenize_words(text);

    // Initialize flat count array
    let vocab_size = config.idf.len();
    if bufs.counts.len() < vocab_size {
        bufs.counts.resize(vocab_size, 0);
    }
    bufs.dirty.clear();

    let ngram_buf = &mut bufs.ngram_buf;
    let counts = &mut bufs.counts;
    let dirty = &mut bufs.dirty;

    let use_hash = config.hash_lookup.is_some();

    for n in config.ngram_min..=config.ngram_max {
        if n > word_ranges.len() {
            break;
        }
        for window in word_ranges.windows(n) {
            ngram_buf.clear();
            if n == 1 {
                let word = &text.as_bytes()[window[0].0..window[0].1];
                for &b in word {
                    ngram_buf.push(b.to_ascii_lowercase());
                }
            } else {
                for (i, range) in window.iter().enumerate() {
                    if i > 0 {
                        ngram_buf.push(b' ');
                    }
                    let word = &text.as_bytes()[range.0..range.1];
                    for &b in word {
                        ngram_buf.push(b.to_ascii_lowercase());
                    }
                }
            }

            let idx_opt = if use_hash {
                config.hash_lookup.as_ref().unwrap().get(ngram_buf)
            } else {
                let ngram_str = unsafe { std::str::from_utf8_unchecked(ngram_buf) };
                config.vocab.get(ngram_str).copied()
            };

            if let Some(idx) = idx_opt {
                if counts[idx] == 0 {
                    dirty.push(idx);
                }
                counts[idx] = counts[idx].saturating_add(1);
            }
        }
    }

    let mut features: SparseVec = Vec::with_capacity(dirty.len());
    for &idx in dirty.iter() {
        let count = counts[idx] as f64;
        counts[idx] = 0;
        let tf = if config.sublinear_tf {
            1.0 + count.ln()
        } else {
            count
        };
        let tfidf = tf * config.idf[idx];
        features.push((idx, tfidf));
    }

    l2_normalize(&mut features);

    features
}

/// Thread-local reusable buffers for char TF-IDF to avoid per-call allocation.
struct CharWbBuffers {
    lower_buf: Vec<u8>,
    padded: Vec<u8>,
    /// Flat count array indexed by vocab index. Avoids hashmap overhead.
    /// Initialized to 80K+ zeros; only non-zero entries are collected.
    counts: Vec<u16>,
    /// Track which indices were set (for sparse cleanup).
    dirty: Vec<usize>,
}

thread_local! {
    static CHAR_WB_BUFS: std::cell::RefCell<CharWbBuffers> = std::cell::RefCell::new(CharWbBuffers {
        lower_buf: Vec::with_capacity(512),
        padded: Vec::with_capacity(128),
        counts: Vec::new(), // Initialized on first use based on vocab size
        dirty: Vec::with_capacity(128),
    });
}

/// Character-level TF-IDF vectorization with word boundaries (char_wb).
///
/// Matches sklearn's TfidfVectorizer with analyzer="char_wb".
/// sklearn's _char_wb_ngrams:
///   1. Normalizes whitespace: `\s+ -> " "`
///   2. Splits on whitespace: `text.split()`
///   3. Pads each token with spaces: `" " + w + " "`
///   4. Extracts char n-grams from each padded token independently
///
/// Uses hash-only lookup table when available for fastest n-gram matching.
pub fn vectorize_char_wb(text: &str, config: &TfidfConfig) -> SparseVec {
    CHAR_WB_BUFS.with(|bufs| {
        let mut bufs = bufs.borrow_mut();
        vectorize_char_wb_inner(text, config, &mut bufs)
    })
}

fn vectorize_char_wb_inner(text: &str, config: &TfidfConfig, bufs: &mut CharWbBuffers) -> SparseVec {
    // Initialize flat count array on first use
    let vocab_size = config.idf.len();
    if bufs.counts.len() < vocab_size {
        bufs.counts.resize(vocab_size, 0);
    }

    // Lowercase into reusable buffer
    let text_bytes = text.as_bytes();
    bufs.lower_buf.clear();
    if text.is_ascii() {
        bufs.lower_buf.extend(text_bytes.iter().map(|b| b.to_ascii_lowercase()));
    } else {
        let lower: String = text.chars().flat_map(|c| c.to_lowercase()).collect();
        bufs.lower_buf.extend_from_slice(lower.as_bytes());
    }

    bufs.dirty.clear();
    let lower_buf = &bufs.lower_buf;
    let padded = &mut bufs.padded;
    let counts = &mut bufs.counts;
    let dirty = &mut bufs.dirty;
    let mut i = 0;
    let len = lower_buf.len();

    if let Some(hl) = &config.hash_lookup {
        while i < len {
            if lower_buf[i].is_ascii_whitespace()
                || (!text.is_ascii() && (lower_buf[i] as char).is_whitespace())
            {
                i += 1;
                continue;
            }
            let word_start = i;
            while i < len && !lower_buf[i].is_ascii_whitespace() {
                i += 1;
            }
            let word_bytes = &lower_buf[word_start..i];

            padded.clear();
            padded.push(b' ');
            padded.extend_from_slice(word_bytes);
            padded.push(b' ');

            let plen = padded.len();
            for n in config.ngram_min..=config.ngram_max {
                if n > plen {
                    break;
                }
                let end = plen - n;
                for start in 0..=end {
                    if let Some(idx) = hl.get(&padded[start..start + n]) {
                        if counts[idx] == 0 {
                            dirty.push(idx);
                        }
                        counts[idx] = counts[idx].saturating_add(1);
                    }
                }
            }
        }
    } else {
        while i < len {
            if lower_buf[i].is_ascii_whitespace()
                || (!text.is_ascii() && (lower_buf[i] as char).is_whitespace())
            {
                i += 1;
                continue;
            }
            let word_start = i;
            while i < len && !lower_buf[i].is_ascii_whitespace() {
                i += 1;
            }
            let word_bytes = &lower_buf[word_start..i];

            padded.clear();
            padded.push(b' ');
            padded.extend_from_slice(word_bytes);
            padded.push(b' ');

            let plen = padded.len();
            for n in config.ngram_min..=config.ngram_max {
                if n > plen {
                    break;
                }
                let end = plen - n;
                for start in 0..=end {
                    let ngram_str =
                        unsafe { std::str::from_utf8_unchecked(&padded[start..start + n]) };
                    if let Some(&idx) = config.vocab.get(ngram_str) {
                        if counts[idx] == 0 {
                            dirty.push(idx);
                        }
                        counts[idx] = counts[idx].saturating_add(1);
                    }
                }
            }
        }
    }

    // Collect features and clean up counts
    let mut features: SparseVec = Vec::with_capacity(dirty.len());
    for &idx in dirty.iter() {
        let count = counts[idx] as f64;
        counts[idx] = 0; // Reset for next call
        let tf = if config.sublinear_tf {
            1.0 + count.ln()
        } else {
            count
        };
        let tfidf = tf * config.idf[idx];
        features.push((idx, tfidf));
    }

    l2_normalize(&mut features);

    features
}

/// L2-normalize a sparse vector in-place.
fn l2_normalize(features: &mut SparseVec) {
    let norm: f64 = features.iter().map(|(_, v)| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for (_, v) in features.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;

    fn make_simple_word_config() -> TfidfConfig {
        let mut vocab = AHashMap::new();
        vocab.insert("hello".to_string(), 0);
        vocab.insert("world".to_string(), 1);
        vocab.insert("hello world".to_string(), 2);
        TfidfConfig {
            vocab,
            idf: vec![1.0, 1.0, 1.0],
            ngram_min: 1,
            ngram_max: 2,
            sublinear_tf: false,

            hash_lookup: None,
        }
    }

    #[test]
    fn test_word_vectorize_basic() {
        let config = make_simple_word_config();
        let features = vectorize_word("hello world", &config);
        assert_eq!(features.len(), 3); // "hello", "world", "hello world"
    }

    #[test]
    fn test_word_vectorize_sublinear() {
        let mut config = make_simple_word_config();
        config.sublinear_tf = true;
        let features = vectorize_word("hello hello world", &config);
        // "hello" appears twice: tf = 1 + ln(2) ≈ 1.693
        let hello_val = features.iter().find(|(i, _)| *i == 0).map(|(_, v)| *v);
        assert!(hello_val.is_some());
    }

    #[test]
    fn test_char_wb_basic() {
        use crate::model::HashLookup;

        let mut vocab = AHashMap::new();
        vocab.insert(" er".to_string(), 0);
        vocab.insert("err".to_string(), 1);
        vocab.insert("rro".to_string(), 2);
        vocab.insert("ror".to_string(), 3);
        vocab.insert("or ".to_string(), 4);

        // Test without hash lookup (fallback path)
        let config_fallback = TfidfConfig {
            vocab: vocab.clone(),
            idf: vec![1.0; 5],
            ngram_min: 3,
            ngram_max: 3,
            sublinear_tf: false,

            hash_lookup: None,
        };
        let features_fallback = vectorize_char_wb("error", &config_fallback);
        assert_eq!(features_fallback.len(), 5);

        // Test with hash lookup (fast path)
        let hl = HashLookup::build(&vocab);
        let config_fast = TfidfConfig {
            vocab,
            idf: vec![1.0; 5],
            ngram_min: 3,
            ngram_max: 3,
            sublinear_tf: false,

            hash_lookup: Some(hl),
        };
        let features_fast = vectorize_char_wb("error", &config_fast);
        assert_eq!(features_fast.len(), 5);

        // Both paths should produce the same features
        let mut sorted_fallback = features_fallback.clone();
        let mut sorted_fast = features_fast.clone();
        sorted_fallback.sort_by_key(|(i, _)| *i);
        sorted_fast.sort_by_key(|(i, _)| *i);
        for ((i1, v1), (i2, v2)) in sorted_fallback.iter().zip(sorted_fast.iter()) {
            assert_eq!(i1, i2);
            assert!((v1 - v2).abs() < 1e-10);
        }
    }
}
