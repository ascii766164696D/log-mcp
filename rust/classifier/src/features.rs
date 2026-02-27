use regex::Regex;

/// Number of handcrafted features.
pub const N_HANDCRAFTED: usize = 17;

/// Extract 17 handcrafted features from a raw log line.
///
/// Features 0-5 are the original features (for backward compatibility).
/// Features 6-16 are new additions for improved recall.
///
/// Matches Python's HandcraftedFeatureExtractor.transform() exactly.
pub fn extract_handcrafted(
    line: &str,
    level_re: &Regex,
    stack_trace_re: &Regex,
    negative_re: Option<&Regex>,
    path_re: Option<&Regex>,
    hex_re: Option<&Regex>,
    has_repeated_punct_pattern: bool,
    key_value_re: Option<&Regex>,
) -> [f64; N_HANDCRAFTED] {
    let bytes = line.as_bytes();
    let length = bytes.len() as f64;
    let denom = if length > 0.0 { length } else { 1.0 };

    let mut n_upper: usize = 0;
    let mut n_digits: usize = 0;
    let mut n_struct_punct: usize = 0;
    let mut n_emphasis: usize = 0;
    // Also compute leading whitespace and repeated punct in the same byte scan
    let mut leading_ws: usize = 0;
    let mut leading_done = false;
    let mut repeated_run: usize = 1;
    let mut has_repeated = false;

    for i in 0..bytes.len() {
        let b = bytes[i];
        if b.is_ascii_uppercase() {
            n_upper += 1;
        }
        if b.is_ascii_digit() {
            n_digits += 1;
        }
        if matches!(b, b':' | b';' | b'=' | b'(' | b')' | b'[' | b']' | b'{' | b'}') {
            n_struct_punct += 1;
        }
        if matches!(b, b'!' | b'?') {
            n_emphasis += 1;
        }
        if !leading_done {
            if b.is_ascii_whitespace() {
                leading_ws += 1;
            } else {
                leading_done = true;
            }
        }
        if i > 0 && !has_repeated {
            if b == bytes[i - 1] {
                repeated_run += 1;
                if repeated_run >= 3 {
                    has_repeated = true;
                }
            } else {
                repeated_run = 1;
            }
        }
    }

    // Original 6 features
    let uppercase_ratio = n_upper as f64 / denom;
    let has_level = if level_re.is_match(line) { 1.0 } else { 0.0 };
    let digit_ratio = n_digits as f64 / denom;
    let has_stack = if stack_trace_re.is_match(line) { 1.0 } else { 0.0 };
    let struct_punct = n_struct_punct as f64 / denom;

    // New features (6-16)
    // Single pass for has_negative + n_negative (avoid scanning twice)
    let (has_negative, n_negative) = match negative_re {
        Some(re) => {
            let count = re.find_iter(line).count();
            (if count > 0 { 1.0 } else { 0.0 }, count as f64)
        }
        None => (0.0, 0.0),
    };

    // Single pass over whitespace-split words for word_count, allcaps, unique_ratio
    let mut word_count: usize = 0;
    let mut n_allcaps: usize = 0;
    let mut seen = ahash::AHashSet::new();
    for w in line.split_ascii_whitespace() {
        word_count += 1;
        // Check allcaps: 3+ chars, all uppercase alphabetic
        if w.len() >= 3 {
            let bytes = w.as_bytes();
            let all_upper_alpha =
                bytes.iter().all(|&b| b.is_ascii_uppercase());
            if all_upper_alpha {
                n_allcaps += 1;
            }
        }
        // For unique ratio, insert lowercase (use bytes for ASCII fast path)
        seen.insert(w.to_ascii_lowercase());
    }

    let leading_ws_f = leading_ws as f64;

    let has_path = match path_re {
        Some(re) => if re.is_match(line) { 1.0 } else { 0.0 },
        None => 0.0,
    };

    let n_hex = match hex_re {
        Some(re) => re.find_iter(line).count() as f64,
        None => 0.0,
    };

    let has_repeated_punct = if has_repeated_punct_pattern && has_repeated {
        1.0
    } else {
        0.0
    };

    let n_kv = match key_value_re {
        Some(re) => re.find_iter(line).count() as f64,
        None => 0.0,
    };

    let unique_ratio = if word_count == 0 {
        0.0
    } else {
        seen.len() as f64 / word_count as f64
    };

    [
        length,             // 0
        uppercase_ratio,    // 1
        has_level,          // 2
        digit_ratio,        // 3
        has_stack,          // 4
        struct_punct,       // 5
        has_negative,       // 6
        n_negative,         // 7
        word_count as f64,  // 8
        n_allcaps as f64,   // 9
        leading_ws_f,       // 10
        has_path,           // 11
        n_hex,              // 12
        has_repeated_punct, // 13
        n_kv,               // 14
        unique_ratio,       // 15
        n_emphasis as f64,  // 16
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_regexes() -> (Regex, Regex, Option<Regex>, Option<Regex>, Option<Regex>, bool, Option<Regex>) {
        let level = Regex::new(r"(?i)ERROR|FATAL|CRITICAL|WARN|EXCEPTION|FAIL").unwrap();
        let stack = Regex::new(r#"(?im)Traceback|^\s+at\s|Caused by:|^\s+File\s""#).unwrap();
        let negative = Some(Regex::new(r"(?i)\b(?:unable|cannot|timeout|denied|refused|unknown|invalid)\b").unwrap());
        let path = Some(Regex::new(r"/[\w./-]{3,}").unwrap());
        let hex = Some(Regex::new(r"\b(?:0x[0-9a-fA-F]+|[0-9a-f]{8,})\b").unwrap());
        let kv = Some(Regex::new(r"\b\w+=\S+").unwrap());
        (level, stack, negative, path, hex, true, kv)
    }

    #[test]
    fn test_error_line() {
        let (level, stack, neg, path, hex, rep, kv) = test_regexes();
        let feats = extract_handcrafted(
            "ERROR: something went wrong",
            &level, &stack, neg.as_ref(), path.as_ref(), hex.as_ref(), rep, kv.as_ref(),
        );
        assert!(feats[0] > 0.0); // length
        assert!(feats[1] > 0.0); // uppercase ratio
        assert_eq!(feats[2], 1.0); // has_level
        assert_eq!(feats[4], 0.0); // no stack trace
    }

    #[test]
    fn test_negative_language() {
        let (level, stack, neg, path, hex, rep, kv) = test_regexes();
        let feats = extract_handcrafted(
            "Connection timeout to database, unable to connect",
            &level, &stack, neg.as_ref(), path.as_ref(), hex.as_ref(), rep, kv.as_ref(),
        );
        assert_eq!(feats[6], 1.0); // has_negative
        assert!(feats[7] >= 2.0); // n_negative (timeout + unable)
    }

    #[test]
    fn test_stack_trace_line() {
        let (level, stack, neg, path, hex, rep, kv) = test_regexes();
        let feats = extract_handcrafted(
            "  at com.example.Main.run(Main.java:42)",
            &level, &stack, neg.as_ref(), path.as_ref(), hex.as_ref(), rep, kv.as_ref(),
        );
        assert_eq!(feats[2], 0.0); // no level keyword
        assert_eq!(feats[4], 1.0); // has stack trace
        assert!(feats[10] > 0.0); // leading whitespace
    }

    #[test]
    fn test_hex_and_path() {
        let (level, stack, neg, path, hex, rep, kv) = test_regexes();
        let feats = extract_handcrafted(
            "iar 00106570 dear 0245abcd /var/log/syslog",
            &level, &stack, neg.as_ref(), path.as_ref(), hex.as_ref(), rep, kv.as_ref(),
        );
        assert!(feats[12] >= 1.0); // n_hex
        assert_eq!(feats[11], 1.0); // has_path
    }

    #[test]
    fn test_empty_line() {
        let (level, stack, neg, path, hex, rep, kv) = test_regexes();
        let feats = extract_handcrafted(
            "",
            &level, &stack, neg.as_ref(), path.as_ref(), hex.as_ref(), rep, kv.as_ref(),
        );
        assert_eq!(feats[0], 0.0); // length 0
        assert_eq!(feats[1], 0.0); // no uppercase
    }
}
