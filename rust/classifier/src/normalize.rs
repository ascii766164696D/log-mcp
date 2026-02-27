use regex::Regex;

use crate::model::{NormalizeRegex, NormalizeRule};

/// Whitespace collapse regex (compiled once).
static WHITESPACE_RE: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// Fast ID pattern replacement.
/// Matches `\b[a-z]+_(?=[a-z]*\d)[a-z0-9]{4,}\b` (case-insensitive)
/// without using fancy_regex lookahead.
///
/// An ID is: one or more letters, underscore, then 4+ alphanumeric chars
/// where the part after `_` contains at least one digit.
fn replace_ids(text: &str) -> String {
    // Quick check: if no underscore, nothing to do (avoids scan)
    if !text.contains('_') {
        return text.to_string();
    }

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut result = Vec::with_capacity(len);
    let mut i = 0;

    while i < len {
        // Look for underscore preceded by [a-zA-Z]+
        if bytes[i] == b'_' && i > 0 {
            // Scan backward to find start of [a-zA-Z]+ prefix
            let mut prefix_start = i;
            while prefix_start > 0 && bytes[prefix_start - 1].is_ascii_alphabetic() {
                prefix_start -= 1;
            }
            // Check word boundary at prefix_start
            let at_word_boundary = prefix_start == 0
                || !bytes[prefix_start - 1].is_ascii_alphanumeric();

            if at_word_boundary && prefix_start < i {
                // Scan forward after underscore for [a-zA-Z0-9]{4,} containing a digit
                let suffix_start = i + 1;
                let mut j = suffix_start;
                let mut has_digit = false;
                while j < len && (bytes[j].is_ascii_alphanumeric()) {
                    if bytes[j].is_ascii_digit() {
                        has_digit = true;
                    }
                    j += 1;
                }
                let suffix_len = j - suffix_start;
                // Check word boundary at end
                let at_end_boundary = j == len || !bytes[j].is_ascii_alphanumeric();

                if suffix_len >= 4 && has_digit && at_end_boundary {
                    // Replace the entire match
                    // Remove the prefix we already wrote
                    let prefix_len = i - prefix_start;
                    result.truncate(result.len() - prefix_len);
                    result.extend_from_slice(b"<ID>");
                    i = j;
                    continue;
                }
            }
        }
        result.push(bytes[i]);
        i += 1;
    }

    // SAFETY: we only manipulate ASCII bytes and the replacement is ASCII
    unsafe { String::from_utf8_unchecked(result) }
}

/// Apply normalization rules in order, then collapse whitespace and trim.
/// This is the Rust port of Python's `normalize_extended()`.
///
/// Uses the fast `regex` crate for most patterns and a custom function
/// for the ID pattern (avoids slow fancy_regex lookahead).
/// Avoids allocation when no patterns match (uses Cow).
pub fn normalize_extended(
    text: &str,
    rules: &[NormalizeRule],
) -> String {
    use std::borrow::Cow;

    let mut result: Cow<str> = Cow::Borrowed(text);
    for rule in rules {
        match &rule.regex {
            NormalizeRegex::Fast(re) => {
                let replaced = re.replace_all(&result, rule.replacement.as_str());
                if let Cow::Owned(s) = replaced {
                    result = Cow::Owned(s);
                }
            }
            NormalizeRegex::Fancy(_) => {
                result = Cow::Owned(replace_ids(&result));
            }
        }
    }
    let collapsed = WHITESPACE_RE.replace_all(&result, " ");
    collapsed.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::NormalizeRule;
    use fancy_regex::Regex as FancyRegex;

    fn make_test_rules() -> Vec<NormalizeRule> {
        vec![
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r"(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}").unwrap()),
                replacement: "<UUID>".to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r"(?i)\b[0-9a-f]{12,}\b").unwrap()),
                replacement: "<HEX>".to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r"0x[0-9a-fA-F]+").unwrap()),
                replacement: "<HEX>".to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b").unwrap()),
                replacement: "<IP>".to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fancy(FancyRegex::new(r"(?i)\b[a-z]+_(?=[a-z]*\d)[a-z0-9]{4,}\b").unwrap()),
                replacement: "<ID>".to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r"\b\d+").unwrap()),
                replacement: "<N>".to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r#""[^"]*""#).unwrap()),
                replacement: r#""<STR>""#.to_string(),
            },
            NormalizeRule {
                regex: NormalizeRegex::Fast(Regex::new(r"'[^']*'").unwrap()),
                replacement: "'<STR>'".to_string(),
            },
        ]
    }

    #[test]
    fn test_uuid_normalization() {
        let rules = make_test_rules();
        let result = normalize_extended(
            "request 550e8400-e29b-41d4-a716-446655440000 failed",
            &rules,
        );
        assert_eq!(result, "request <UUID> failed");
    }

    #[test]
    fn test_ip_normalization() {
        let rules = make_test_rules();
        let result = normalize_extended("connection from 192.168.1.100 refused", &rules);
        assert_eq!(result, "connection from <IP> refused");
    }

    #[test]
    fn test_number_normalization() {
        let rules = make_test_rules();
        let result = normalize_extended("processed 12345 items in 67 seconds", &rules);
        assert_eq!(result, "processed <N> items in <N> seconds");
    }

    #[test]
    fn test_whitespace_collapse() {
        let rules = make_test_rules();
        let result = normalize_extended("  too   much   space  ", &rules);
        assert_eq!(result, "too much space");
    }

    #[test]
    fn test_id_with_lookahead() {
        let rules = make_test_rules();
        let result = normalize_extended("task task_abc123 completed", &rules);
        assert_eq!(result, "task <ID> completed");
    }
}
