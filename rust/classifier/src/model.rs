use std::fs;
use std::hash::{BuildHasher, Hash, Hasher};
use std::path::Path;

use ahash::AHashMap;
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde::Deserialize;

/// Normalization rule that uses whichever regex engine is needed.
/// Most patterns work with the fast `regex` crate; only patterns with
/// lookahead/lookbehind need the slower `fancy_regex`.
#[allow(dead_code)]
pub enum NormalizeRegex {
    Fast(Regex),
    Fancy(FancyRegex),
}

/// Raw JSON structure for a TF-IDF config.
#[derive(Deserialize)]
pub struct TfidfConfigRaw {
    pub vocab: AHashMap<String, usize>,
    pub idf: Vec<f64>,
    pub ngram_range: (usize, usize),
    #[serde(default)]
    pub sublinear_tf: bool,
}

/// Raw JSON structure for a normalization pattern.
#[derive(Deserialize)]
pub struct NormalizePatternRaw {
    pub pattern: String,
    pub replacement: String,
    #[serde(default)]
    pub case_insensitive: bool,
}

/// Raw JSON structure for handcrafted feature config.
#[derive(Deserialize)]
pub struct HandcraftedConfigRaw {
    pub level_pattern: String,
    pub stack_trace_pattern: String,
    #[serde(default)]
    pub negative_pattern: Option<String>,
    #[serde(default)]
    pub path_pattern: Option<String>,
    #[serde(default)]
    pub hex_pattern: Option<String>,
    #[serde(default)]
    pub repeated_punct_pattern: Option<String>,
    #[serde(default)]
    pub key_value_pattern: Option<String>,
}

/// Raw JSON structure for classifier weights.
#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ClassifierWeightsRaw {
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub classes: Vec<String>,
    pub n_word_features: usize,
    pub n_char_features: usize,
    pub n_handcrafted_features: usize,
}

/// Top-level raw JSON structure.
#[derive(Deserialize)]
pub struct ModelRaw {
    pub word_tfidf: TfidfConfigRaw,
    pub char_tfidf: TfidfConfigRaw,
    pub normalize_patterns: Vec<NormalizePatternRaw>,
    pub handcrafted: HandcraftedConfigRaw,
    pub classifier: ClassifierWeightsRaw,
}

/// A hash-only lookup table for fast n-gram matching.
/// Uses open addressing with ahash. No string comparison needed when
/// hash collisions are verified absent at build time.
pub struct HashLookup {
    /// (hash, vocab_index) pairs in open-addressing slots.
    /// Empty slots have hash = 0 and vocab_index = usize::MAX.
    table: Vec<(u64, usize)>,
    mask: u64,
    hasher: ahash::RandomState,
}

impl HashLookup {
    /// Build a hash lookup table from a vocab map.
    /// Panics if there are hash collisions (extremely unlikely with 64-bit ahash).
    pub fn build(vocab: &AHashMap<String, usize>) -> Self {
        // Size: next power of 2, at least 2x entries for low collision rate
        let capacity = (vocab.len() * 2).next_power_of_two();
        let mask = (capacity - 1) as u64;
        let mut table = vec![(0u64, usize::MAX); capacity];

        let hasher = ahash::RandomState::with_seeds(0, 0, 0, 0);

        for (ngram, &idx) in vocab {
            let mut h = hasher.build_hasher();
            ngram.as_bytes().hash(&mut h);
            let hash = h.finish();
            // Ensure hash is never 0 (reserved for empty)
            let hash = if hash == 0 { 1 } else { hash };

            let mut slot = (hash & mask) as usize;
            loop {
                if table[slot].0 == 0 {
                    table[slot] = (hash, idx);
                    break;
                }
                if table[slot].0 == hash {
                    panic!(
                        "Hash collision in char vocab (hash={hash:#x}). \
                         This is extremely unlikely with 64-bit hashes."
                    );
                }
                slot = ((slot + 1) as u64 & mask) as usize;
            }
        }

        HashLookup { table, mask, hasher }
    }

    /// Look up a byte slice. Returns Some(vocab_index) if found.
    #[inline]
    pub fn get(&self, bytes: &[u8]) -> Option<usize> {
        let mut h = self.hasher.build_hasher();
        bytes.hash(&mut h);
        let hash = h.finish();
        let hash = if hash == 0 { 1 } else { hash };

        let mut slot = (hash & self.mask) as usize;
        loop {
            let entry = unsafe { self.table.get_unchecked(slot) };
            if entry.0 == hash {
                return Some(entry.1);
            }
            if entry.0 == 0 {
                return None;
            }
            slot = ((slot + 1) as u64 & self.mask) as usize;
        }
    }
}

/// Compiled TF-IDF config ready for inference.
pub struct TfidfConfig {
    pub vocab: AHashMap<String, usize>,
    pub idf: Vec<f64>,
    pub ngram_min: usize,
    pub ngram_max: usize,
    pub sublinear_tf: bool,
    /// Hash-only lookup table for fast char n-gram matching.
    pub hash_lookup: Option<HashLookup>,
}

/// Compiled normalization rule.
pub struct NormalizeRule {
    pub regex: NormalizeRegex,
    pub replacement: String,
}

/// Full compiled model ready for inference.
pub struct Model {
    pub word_tfidf: TfidfConfig,
    pub char_tfidf: TfidfConfig,
    pub normalize_rules: Vec<NormalizeRule>,
    pub level_regex: Regex,
    pub stack_trace_regex: Regex,
    pub negative_regex: Option<Regex>,
    pub path_regex: Option<Regex>,
    pub hex_regex: Option<Regex>,
    pub has_repeated_punct_pattern: bool,
    pub key_value_regex: Option<Regex>,
    pub coef: Vec<f64>,
    pub intercept: f64,
    pub n_word_features: usize,
    pub n_char_features: usize,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read model file: {e}"))?;
        let raw: ModelRaw = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse model JSON: {e}"))?;

        // Compile normalization patterns.
        // Try the fast `regex` crate first; fall back to `fancy_regex` for
        // patterns that use lookahead/lookbehind (only the ID pattern needs it).
        let normalize_rules: Vec<NormalizeRule> = raw
            .normalize_patterns
            .into_iter()
            .map(|p| {
                let pattern = if p.case_insensitive {
                    format!("(?i){}", p.pattern)
                } else {
                    p.pattern
                };
                let regex = match Regex::new(&pattern) {
                    Ok(r) => NormalizeRegex::Fast(r),
                    Err(_) => {
                        let r = FancyRegex::new(&pattern)
                            .map_err(|e| format!("Invalid normalize regex: {e}"))
                            .unwrap();
                        NormalizeRegex::Fancy(r)
                    }
                };
                NormalizeRule {
                    regex,
                    replacement: p.replacement,
                }
            })
            .collect();

        // Compile handcrafted feature regexes
        let level_regex = Regex::new(&format!("(?i){}", raw.handcrafted.level_pattern))
            .map_err(|e| format!("Invalid level regex: {e}"))?;
        let stack_trace_regex =
            Regex::new(&format!("(?im){}", raw.handcrafted.stack_trace_pattern))
                .map_err(|e| format!("Invalid stack trace regex: {e}"))?;

        let compile_optional = |pat: &Option<String>, flags: &str| -> Result<Option<Regex>, String> {
            match pat {
                Some(p) => Regex::new(&format!("{flags}{p}"))
                    .map(Some)
                    .map_err(|e| format!("Invalid regex: {e}")),
                None => Ok(None),
            }
        };
        let negative_regex = compile_optional(&raw.handcrafted.negative_pattern, "(?i)")?;
        let path_regex = compile_optional(&raw.handcrafted.path_pattern, "")?;
        let hex_regex = compile_optional(&raw.handcrafted.hex_pattern, "")?;
        let has_repeated_punct_pattern = raw.handcrafted.repeated_punct_pattern.is_some();
        let key_value_regex = compile_optional(&raw.handcrafted.key_value_pattern, "")?;

        let n_word = raw.classifier.n_word_features;
        let n_char = raw.classifier.n_char_features;
        let n_hc = raw.classifier.n_handcrafted_features;
        let expected = n_word + n_char + n_hc;
        if raw.classifier.coef.len() != expected {
            return Err(format!(
                "Coefficient dimension mismatch: {} != {}",
                raw.classifier.coef.len(),
                expected
            ));
        }

        // Build hash lookup for char TF-IDF (fast n-gram matching without string comparison)
        let char_hash_lookup = HashLookup::build(&raw.char_tfidf.vocab);

        Ok(Model {
            word_tfidf: TfidfConfig {
                hash_lookup: Some(HashLookup::build(&raw.word_tfidf.vocab)),
                vocab: raw.word_tfidf.vocab,
                idf: raw.word_tfidf.idf,
                ngram_min: raw.word_tfidf.ngram_range.0,
                ngram_max: raw.word_tfidf.ngram_range.1,
                sublinear_tf: raw.word_tfidf.sublinear_tf,
            },
            char_tfidf: TfidfConfig {
                vocab: raw.char_tfidf.vocab,
                idf: raw.char_tfidf.idf,
                ngram_min: raw.char_tfidf.ngram_range.0,
                ngram_max: raw.char_tfidf.ngram_range.1,
                sublinear_tf: raw.char_tfidf.sublinear_tf,
                hash_lookup: Some(char_hash_lookup),
            },
            normalize_rules,
            level_regex,
            stack_trace_regex,
            negative_regex,
            path_regex,
            hex_regex,
            has_repeated_punct_pattern,
            key_value_regex,
            coef: raw.classifier.coef,
            intercept: raw.classifier.intercept,
            n_word_features: n_word,
            n_char_features: n_char,
        })
    }
}
