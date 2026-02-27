mod classifier;
mod features;
mod model;
mod normalize;
mod tfidf;
#[cfg(feature = "transformer")]
mod transformer;

use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use classifier::Label;
use model::Model;

#[pyclass]
struct LookSkipClassifier {
    model: Model,
}

#[pymethods]
impl LookSkipClassifier {
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        let model = Model::load(Path::new(model_path))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(LookSkipClassifier { model })
    }

    /// Classify a single line. Returns (label_str, probability).
    fn classify_line(&self, line: &str, threshold: f64) -> (String, f64) {
        let (label, prob) = self.model.classify_line(line, threshold);
        let label_str = match label {
            Label::Look => "LOOK",
            Label::Skip => "SKIP",
        };
        (label_str.to_string(), prob)
    }

    /// Classify a batch of lines. Returns list of (label_str, probability).
    fn classify_batch(&self, lines: Vec<String>, threshold: f64) -> Vec<(String, f64)> {
        lines
            .iter()
            .map(|line| {
                let (label, prob) = self.model.classify_line(line, threshold);
                let label_str = match label {
                    Label::Look => "LOOK",
                    Label::Skip => "SKIP",
                };
                (label_str.to_string(), prob)
            })
            .collect()
    }

    /// Classify an entire file. Returns a dict with results.
    fn classify_file(
        &self,
        py: Python<'_>,
        file_path: &str,
        threshold: f64,
        max_lines: usize,
        max_look_lines: usize,
    ) -> PyResult<PyObject> {
        let result = py
            .allow_threads(|| {
                self.model
                    .classify_file(Path::new(file_path), threshold, max_lines, max_look_lines)
            })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        let dict = PyDict::new(py);
        dict.set_item("total_lines", result.total_lines)?;
        dict.set_item("look_count", result.look_count)?;
        dict.set_item("skip_count", result.skip_count)?;
        dict.set_item("processing_time_s", result.processing_time_s)?;
        dict.set_item("lines_per_second", result.lines_per_second)?;

        let look_lines = PyList::empty(py);
        for (line_no, prob, text) in &result.look_lines {
            let tuple = (*line_no, *prob, text.as_str());
            look_lines.append(tuple)?;
        }
        dict.set_item("look_lines", look_lines)?;

        Ok(dict.into())
    }

    /// Profile classification: time each step (single-threaded). Returns dict with ns timings.
    #[pyo3(signature = (file_path, max_lines=0))]
    fn profile_file(
        &self,
        py: Python<'_>,
        file_path: &str,
        max_lines: usize,
    ) -> PyResult<PyObject> {
        let (total, t_norm, t_word, t_char, t_hc, t_dot) = py
            .allow_threads(|| {
                self.model.profile_file(Path::new(file_path), max_lines)
            })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        let dict = PyDict::new(py);
        dict.set_item("total_lines", total)?;
        dict.set_item("normalize_ns", t_norm as u64)?;
        dict.set_item("word_tfidf_ns", t_word as u64)?;
        dict.set_item("char_tfidf_ns", t_char as u64)?;
        dict.set_item("handcrafted_ns", t_hc as u64)?;
        dict.set_item("dot_product_ns", t_dot as u64)?;
        Ok(dict.into())
    }

    /// Classify a file with keyword analysis in a single parallel pass.
    /// Returns a dict with classification results + error/warn capture stats.
    fn classify_file_with_keywords(
        &self,
        py: Python<'_>,
        file_path: &str,
        threshold: f64,
        max_lines: usize,
        max_look_lines: usize,
    ) -> PyResult<PyObject> {
        let result = py
            .allow_threads(|| {
                self.model
                    .classify_file_with_keywords(
                        Path::new(file_path), threshold, max_lines, max_look_lines,
                    )
            })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        let dict = PyDict::new(py);
        dict.set_item("total_lines", result.base.total_lines)?;
        dict.set_item("look_count", result.base.look_count)?;
        dict.set_item("skip_count", result.base.skip_count)?;
        dict.set_item("processing_time_s", result.base.processing_time_s)?;
        dict.set_item("lines_per_second", result.base.lines_per_second)?;
        dict.set_item("error_lines", result.error_lines)?;
        dict.set_item("error_captured", result.error_captured)?;
        dict.set_item("warn_lines", result.warn_lines)?;
        dict.set_item("warn_captured", result.warn_captured)?;

        let look_lines = PyList::empty(py);
        for (line_no, prob, text) in &result.base.look_lines {
            let tuple = (*line_no, *prob, text.as_str());
            look_lines.append(tuple)?;
        }
        dict.set_item("look_lines", look_lines)?;

        Ok(dict.into())
    }
}

/// Transformer-based LOOK/SKIP classifier (BERT-mini with candle).
#[cfg(feature = "transformer")]
#[pyclass]
struct TransformerClassifier {
    model: transformer::TransformerModel,
}

#[cfg(feature = "transformer")]
#[pymethods]
impl TransformerClassifier {
    #[new]
    fn new(model_dir: &str) -> PyResult<Self> {
        let model = transformer::TransformerModel::load(Path::new(model_dir))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(TransformerClassifier { model })
    }

    /// Classify a single line. Returns (label_str, probability).
    fn classify_line(&self, line: &str, threshold: f64) -> (String, f64) {
        let (is_look, prob) = self.model.classify_line(line, threshold);
        let label_str = if is_look { "LOOK" } else { "SKIP" };
        (label_str.to_string(), prob)
    }

    /// Classify a batch of lines. Returns list of (label_str, probability).
    fn classify_batch(
        &self,
        lines: Vec<String>,
        threshold: f64,
    ) -> PyResult<Vec<(String, f64)>> {
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let results = self
            .model
            .classify_batch(&refs, threshold)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        Ok(results
            .into_iter()
            .map(|(is_look, prob)| {
                let label = if is_look { "LOOK" } else { "SKIP" };
                (label.to_string(), prob)
            })
            .collect())
    }

    /// Classify an entire file. Returns a dict with results.
    #[pyo3(signature = (file_path, threshold=0.5, max_lines=0, max_look_lines=200, batch_size=128))]
    fn classify_file(
        &self,
        py: Python<'_>,
        file_path: &str,
        threshold: f64,
        max_lines: usize,
        max_look_lines: usize,
        batch_size: usize,
    ) -> PyResult<PyObject> {
        let result = py
            .allow_threads(|| {
                self.model.classify_file(
                    Path::new(file_path),
                    threshold,
                    max_lines,
                    max_look_lines,
                    batch_size,
                )
            })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        let dict = PyDict::new(py);
        dict.set_item("total_lines", result.total_lines)?;
        dict.set_item("look_count", result.look_count)?;
        dict.set_item("skip_count", result.skip_count)?;
        dict.set_item("processing_time_s", result.processing_time_s)?;
        dict.set_item("lines_per_second", result.lines_per_second)?;

        let look_lines = PyList::empty(py);
        for (line_no, prob, text) in &result.look_lines {
            let tuple = (*line_no, *prob, text.as_str());
            look_lines.append(tuple)?;
        }
        dict.set_item("look_lines", look_lines)?;

        Ok(dict.into())
    }
}

/// Python module definition.
#[pymodule]
fn look_skip_classifier(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LookSkipClassifier>()?;
    #[cfg(feature = "transformer")]
    m.add_class::<TransformerClassifier>()?;
    Ok(())
}
