use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use rayon::prelude::*;
use regex::Regex;

use crate::features::extract_handcrafted;
use crate::model::Model;
use crate::normalize::normalize_extended;
use crate::tfidf::{vectorize_char_wb, vectorize_word, SparseVec};

/// Classification label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Look,
    Skip,
}

/// Aggregated result of classifying a file.
pub struct FileResult {
    pub total_lines: usize,
    pub look_count: usize,
    pub skip_count: usize,
    pub look_lines: Vec<(usize, f64, String)>, // (line_number, probability, text)
    pub processing_time_s: f64,
    pub lines_per_second: f64,
}

/// Extended result with keyword analysis.
pub struct FileResultWithKeywords {
    pub base: FileResult,
    pub error_lines: usize,
    pub error_captured: usize,
    pub warn_lines: usize,
    pub warn_captured: usize,
}

/// Compiled keyword regexes (lazily initialized).
static ERROR_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r"(?i)\bERROR\b|\bFATAL\b|\bCRITICAL\b|\bSEVERE\b|\bFAILURE\b|\bFAILED\b|\bException\b|\berror\b").unwrap()
});
static WARN_RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r"(?i)\bWARN\b|\bWARNING\b").unwrap()
});

/// Read lines from a BufReader with lossy UTF-8 conversion (like Python's errors="replace").
fn read_lines_lossy<R: std::io::Read>(reader: BufReader<R>) -> impl Iterator<Item = String> {
    LossyLineIterator { reader, buf: Vec::new(), done: false }
}

struct LossyLineIterator<R: std::io::Read> {
    reader: BufReader<R>,
    buf: Vec<u8>,
    done: bool,
}

impl<R: std::io::Read> Iterator for LossyLineIterator<R> {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        if self.done {
            return None;
        }
        self.buf.clear();
        match self.reader.read_until(b'\n', &mut self.buf) {
            Ok(0) => {
                self.done = true;
                None
            }
            Ok(_) => {
                // Strip trailing newline
                if self.buf.last() == Some(&b'\n') {
                    self.buf.pop();
                    if self.buf.last() == Some(&b'\r') {
                        self.buf.pop();
                    }
                }
                Some(String::from_utf8_lossy(&self.buf).into_owned())
            }
            Err(_) => {
                self.done = true;
                None
            }
        }
    }
}

impl Model {
    /// Classify a single line. Returns (label, P(LOOK)).
    ///
    /// sklearn binary LR convention with classes_ = ['LOOK', 'SKIP']:
    /// coef_ is for the last class (SKIP), so:
    ///   score = coef . x + intercept
    ///   P(SKIP) = sigmoid(score) = 1/(1+exp(-score))
    ///   P(LOOK) = 1 - P(SKIP) = 1/(1+exp(score))
    pub fn classify_line(&self, raw_line: &str, threshold: f64) -> (Label, f64) {
        // 1. Normalize for word TF-IDF
        let normalized = normalize_extended(raw_line, &self.normalize_rules);

        // 2. Word TF-IDF on normalized text
        let word_features = vectorize_word(&normalized, &self.word_tfidf);

        // 3. Char TF-IDF on raw text (sklearn uses raw input for char features)
        let char_features = vectorize_char_wb(raw_line, &self.char_tfidf);

        // 4. Handcrafted features on raw text
        let handcrafted = extract_handcrafted(
            raw_line,
            &self.level_regex,
            &self.stack_trace_regex,
            self.negative_regex.as_ref(),
            self.path_regex.as_ref(),
            self.hex_regex.as_ref(),
            self.has_repeated_punct_pattern,
            self.key_value_regex.as_ref(),
        );

        // 5. Dot product
        let score = self.dot_product(&word_features, &char_features, &handcrafted);

        // 6. P(LOOK) = sigmoid(-score) = 1/(1+exp(score))
        let p_look = 1.0 / (1.0 + score.exp());

        let label = if p_look >= threshold {
            Label::Look
        } else {
            Label::Skip
        };

        (label, p_look)
    }

    /// Compute dot product of feature vector with classifier coefficients.
    ///
    /// coef layout: [word_features | char_features | handcrafted_features]
    fn dot_product(
        &self,
        word_features: &SparseVec,
        char_features: &SparseVec,
        handcrafted: &[f64; crate::features::N_HANDCRAFTED],
    ) -> f64 {
        let mut score = self.intercept;

        // Word features: coef[0..n_word]
        for &(idx, val) in word_features {
            score += self.coef[idx] * val;
        }

        // Char features: coef[n_word..n_word+n_char]
        let char_offset = self.n_word_features;
        for &(idx, val) in char_features {
            score += self.coef[char_offset + idx] * val;
        }

        // Handcrafted features: coef[n_word+n_char..]
        let hc_offset = self.n_word_features + self.n_char_features;
        for (i, &val) in handcrafted.iter().enumerate() {
            score += self.coef[hc_offset + i] * val;
        }

        score
    }

    /// Profile classification: time each step across many lines (single-threaded).
    /// Returns (total_lines, normalize_ns, word_tfidf_ns, char_tfidf_ns, handcrafted_ns, dot_product_ns).
    pub fn profile_file(
        &self,
        path: &Path,
        max_lines: usize,
    ) -> Result<(usize, u128, u128, u128, u128, u128), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open file: {e}"))?;
        let reader = BufReader::with_capacity(1024 * 1024, file);
        let limit = if max_lines == 0 { usize::MAX } else { max_lines };

        let mut total = 0usize;
        let mut t_normalize = 0u128;
        let mut t_word_tfidf = 0u128;
        let mut t_char_tfidf = 0u128;
        let mut t_handcrafted = 0u128;
        let mut t_dot = 0u128;

        for line in read_lines_lossy(reader) {
            if total >= limit { break; }
            total += 1;

            let t0 = Instant::now();
            let normalized = normalize_extended(&line, &self.normalize_rules);
            t_normalize += t0.elapsed().as_nanos();

            let t1 = Instant::now();
            let word_features = vectorize_word(&normalized, &self.word_tfidf);
            t_word_tfidf += t1.elapsed().as_nanos();

            let t2 = Instant::now();
            let char_features = vectorize_char_wb(&line, &self.char_tfidf);
            t_char_tfidf += t2.elapsed().as_nanos();

            let t3 = Instant::now();
            let handcrafted = extract_handcrafted(
                &line,
                &self.level_regex,
                &self.stack_trace_regex,
                self.negative_regex.as_ref(),
                self.path_regex.as_ref(),
                self.hex_regex.as_ref(),
                self.has_repeated_punct_pattern,
                self.key_value_regex.as_ref(),
            );
            t_handcrafted += t3.elapsed().as_nanos();

            let t4 = Instant::now();
            let _score = self.dot_product(&word_features, &char_features, &handcrafted);
            t_dot += t4.elapsed().as_nanos();
        }

        Ok((total, t_normalize, t_word_tfidf, t_char_tfidf, t_handcrafted, t_dot))
    }

    /// Maximum bytes to read per chunk (128 MB).
    /// Ensures large files don't exhaust memory.
    const CHUNK_BYTES: usize = 128 * 1024 * 1024;

    /// Read lines into chunks on a background thread, sending them via channel.
    /// The classifier processes chunk N while the reader fills chunk N+1.
    fn read_chunks_pipelined(
        path: &Path,
        limit: usize,
    ) -> Result<(mpsc::Receiver<(Vec<String>, usize)>, std::thread::JoinHandle<()>), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open file: {e}"))?;
        let reader = BufReader::with_capacity(2 * 1024 * 1024, file);

        let (tx, rx) = mpsc::sync_channel(2); // Buffer 2 chunks for pipelining

        let handle = std::thread::spawn(move || {
            let mut total: usize = 0;
            let mut chunk: Vec<String> = Vec::new();
            let mut chunk_bytes: usize = 0;

            for line in read_lines_lossy(reader) {
                if total >= limit {
                    break;
                }
                chunk_bytes += line.len();
                chunk.push(line);
                total += 1;

                if chunk_bytes >= Self::CHUNK_BYTES || total >= limit {
                    let base = total - chunk.len();
                    if tx.send((std::mem::take(&mut chunk), base)).is_err() {
                        return;
                    }
                    chunk_bytes = 0;
                }
            }

            if !chunk.is_empty() {
                let base = total - chunk.len();
                let _ = tx.send((chunk, base));
            }
        });

        Ok((rx, handle))
    }

    /// Classify all lines in a file using pipelined I/O + parallel classification.
    ///
    /// A background thread reads lines into 128 MB chunks. The main thread
    /// classifies each chunk in parallel with rayon while the next chunk is
    /// being read. At most 2 chunks (~256 MB) are in memory at once.
    pub fn classify_file(
        &self,
        path: &Path,
        threshold: f64,
        max_lines: usize,
        max_look_lines: usize,
    ) -> Result<FileResult, String> {
        let limit = if max_lines == 0 { usize::MAX } else { max_lines };
        let (rx, handle) = Self::read_chunks_pipelined(path, limit)?;

        let start = Instant::now();
        let mut total_lines: usize = 0;
        let mut look_count: usize = 0;
        let mut skip_count: usize = 0;
        let mut look_lines: Vec<(usize, f64, String)> = Vec::new();

        for (chunk, base_line_no) in rx {
            total_lines += chunk.len();

            let results: Vec<(Label, f64)> = chunk
                .par_iter()
                .map(|line| self.classify_line(line, threshold))
                .collect();

            for (i, (label, prob)) in results.iter().enumerate() {
                match label {
                    Label::Look => {
                        look_count += 1;
                        if look_lines.len() < max_look_lines {
                            look_lines.push((base_line_no + i + 1, *prob, chunk[i].clone()));
                        }
                    }
                    Label::Skip => {
                        skip_count += 1;
                    }
                }
            }
        }

        handle.join().ok();
        let elapsed = start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            total_lines as f64 / elapsed
        } else {
            0.0
        };

        Ok(FileResult {
            total_lines,
            look_count,
            skip_count,
            look_lines,
            processing_time_s: elapsed,
            lines_per_second: rate,
        })
    }

    /// Classify all lines in a file with keyword analysis.
    ///
    /// Uses pipelined I/O + parallel classification with ERROR/WARN counting.
    pub fn classify_file_with_keywords(
        &self,
        path: &Path,
        threshold: f64,
        max_lines: usize,
        max_look_lines: usize,
    ) -> Result<FileResultWithKeywords, String> {
        let limit = if max_lines == 0 { usize::MAX } else { max_lines };
        let (rx, handle) = Self::read_chunks_pipelined(path, limit)?;

        let start = Instant::now();
        let mut total_lines: usize = 0;
        let mut look_count: usize = 0;
        let mut skip_count: usize = 0;
        let mut look_lines: Vec<(usize, f64, String)> = Vec::new();
        let mut error_lines: usize = 0;
        let mut error_captured: usize = 0;
        let mut warn_lines: usize = 0;
        let mut warn_captured: usize = 0;

        for (chunk, base_line_no) in rx {
            total_lines += chunk.len();

            let results: Vec<(Label, f64, bool, bool)> = chunk
                .par_iter()
                .map(|line| {
                    let (label, prob) = self.classify_line(line, threshold);
                    let is_error = ERROR_RE.is_match(line);
                    let is_warn = !is_error && WARN_RE.is_match(line);
                    (label, prob, is_error, is_warn)
                })
                .collect();

            for (i, (label, prob, is_error, is_warn)) in results.iter().enumerate() {
                let is_look = *label == Label::Look;
                match label {
                    Label::Look => {
                        look_count += 1;
                        if look_lines.len() < max_look_lines {
                            look_lines.push((base_line_no + i + 1, *prob, chunk[i].clone()));
                        }
                    }
                    Label::Skip => {
                        skip_count += 1;
                    }
                }
                if *is_error {
                    error_lines += 1;
                    if is_look { error_captured += 1; }
                } else if *is_warn {
                    warn_lines += 1;
                    if is_look { warn_captured += 1; }
                }
            }
        }

        handle.join().ok();
        let elapsed = start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 { total_lines as f64 / elapsed } else { 0.0 };

        Ok(FileResultWithKeywords {
            base: FileResult {
                total_lines,
                look_count,
                skip_count,
                look_lines,
                processing_time_s: elapsed,
                lines_per_second: rate,
            },
            error_lines,
            error_captured,
            warn_lines,
            warn_captured,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/models/look_skip_model.json")
    }

    fn load_model() -> Model {
        let path = model_path();
        Model::load(&path).unwrap_or_else(|e| {
            panic!("Failed to load model from {}: {e}", path.display())
        })
    }

    #[test]
    fn test_model_loads() {
        let model = load_model();
        assert!(model.coef.len() > 0);
        assert!(model.n_word_features > 0);
        assert!(model.n_char_features > 0);
    }

    #[test]
    fn test_classify_error_lines_as_look() {
        let model = load_model();
        let error_lines = [
            "ERROR: connection refused to database server",
            "FATAL: out of memory, cannot allocate 1024 bytes",
            "java.lang.NullPointerException: null",
            "  at com.example.Service.handle(Service.java:42)",
            "Traceback (most recent call last):",
            "CRITICAL: disk /dev/sda1 is 99% full",
            "kernel: segfault at 0000000000000000",
            "Permission denied: /etc/shadow",
        ];
        for line in &error_lines {
            let (label, prob) = model.classify_line(line, 0.5);
            assert_eq!(
                label,
                Label::Look,
                "Expected LOOK for {line:?}, got SKIP (p_look={prob:.3})"
            );
        }
    }

    #[test]
    fn test_classify_routine_lines_as_skip() {
        let model = load_model();
        let routine_lines = [
            "2024-01-15 10:30:45 INFO  Server started on port 8080",
            "2024-01-15 10:30:46 DEBUG Processing request GET /api/health",
            "INFO: Heartbeat check passed",
            "Compiling dependencies... done",
            "Downloaded 150 packages in 2.3s",
        ];
        for line in &routine_lines {
            let (label, prob) = model.classify_line(line, 0.5);
            assert_eq!(
                label,
                Label::Skip,
                "Expected SKIP for {line:?}, got LOOK (p_look={prob:.3})"
            );
        }
    }

    #[test]
    fn test_classify_file_end_to_end() {
        let model = load_model();

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "INFO: starting up").unwrap();
        writeln!(tmp, "DEBUG: loaded config").unwrap();
        writeln!(tmp, "ERROR: connection refused").unwrap();
        writeln!(tmp, "INFO: retrying in 5s").unwrap();
        writeln!(tmp, "FATAL: giving up after 3 retries").unwrap();
        tmp.flush().unwrap();

        let result = model
            .classify_file(tmp.path(), 0.5, 0, 200)
            .unwrap();

        assert_eq!(result.total_lines, 5);
        assert_eq!(result.look_count + result.skip_count, 5);
        assert!(result.look_count >= 2, "Expected at least 2 LOOK lines, got {}", result.look_count);
        assert!(result.processing_time_s >= 0.0);
    }

    #[test]
    fn test_classify_file_with_keywords() {
        let model = load_model();

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "INFO: all good").unwrap();
        writeln!(tmp, "ERROR: disk full").unwrap();
        writeln!(tmp, "WARNING: memory usage high").unwrap();
        writeln!(tmp, "FATAL: cannot continue").unwrap();
        tmp.flush().unwrap();

        let result = model
            .classify_file_with_keywords(tmp.path(), 0.5, 0, 200)
            .unwrap();

        assert_eq!(result.base.total_lines, 4);
        assert!(result.error_lines >= 2, "Expected at least 2 error keyword lines, got {}", result.error_lines);
        assert!(result.warn_lines >= 1, "Expected at least 1 warn keyword line, got {}", result.warn_lines);
    }
}
