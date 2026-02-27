//! BERT-mini transformer classifier for LOOK/SKIP log line classification.
//!
//! Uses candle for inference with optional Metal (Apple Silicon GPU) acceleration.
//! Loads a fine-tuned BertForSequenceClassification from safetensors format.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use tokenizers::Tokenizer;

/// BERT + pooler + classifier head.
///
/// Reimplements PyTorch's `BertForSequenceClassification`:
///   hidden_states = bert(input_ids, token_type_ids, attention_mask)
///   pooled = tanh(pooler_dense(hidden_states[:, 0]))  # CLS token
///   logits = classifier(pooled)
struct BertClassifier {
    bert: BertModel,
    pooler_dense: Linear,
    classifier: Linear,
}

impl BertClassifier {
    fn load(vb: VarBuilder, config: &BertConfig, num_labels: usize) -> candle_core::Result<Self> {
        let bert = BertModel::load(vb.clone(), config)?;
        let pooler_dense = candle_nn::linear(
            config.hidden_size,
            config.hidden_size,
            vb.pp("bert.pooler.dense"),
        )?;
        let classifier = candle_nn::linear(config.hidden_size, num_labels, vb.pp("classifier"))?;
        Ok(Self {
            bert,
            pooler_dense,
            classifier,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        // BERT encoder: (batch, seq_len, hidden_size)
        let hidden_states = self.bert.forward(input_ids, token_type_ids, attention_mask)?;

        // Pooler: take CLS token (index 0), dense + tanh
        // contiguous() is needed because narrow creates a non-contiguous view
        let cls_output = hidden_states.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?;
        let pooled = self.pooler_dense.forward(&cls_output)?.tanh()?;

        // Classifier: (batch, hidden_size) -> (batch, num_labels)
        self.classifier.forward(&pooled)
    }
}

/// High-level transformer classifier for LOOK/SKIP classification.
pub struct TransformerModel {
    model: BertClassifier,
    tokenizer: Tokenizer,
    device: Device,
    max_seq_len: usize,
}

/// Result of classifying a file with the transformer.
pub struct TransformerFileResult {
    pub total_lines: usize,
    pub look_count: usize,
    pub skip_count: usize,
    pub look_lines: Vec<(usize, f64, String)>,
    pub processing_time_s: f64,
    pub lines_per_second: f64,
}

impl TransformerModel {
    /// Load model from a directory containing config.json, model.safetensors, tokenizer.json.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        // Select device: Metal > CPU
        let device = Self::select_device();
        let device_name = match &device {
            Device::Cpu => "CPU",
            _ => "Metal GPU",
        };

        // Load config
        let config_path = model_dir.join("config.json");
        let config_str =
            std::fs::read_to_string(&config_path).map_err(|e| format!("Config: {e}"))?;
        let config: BertConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("Config parse: {e}"))?;

        // Load weights
        let weights_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                .map_err(|e| format!("Weights: {e}"))?
        };

        // Load model (2 labels: LOOK=0, SKIP=1)
        let model =
            BertClassifier::load(vb, &config, 2).map_err(|e| format!("Model load: {e}"))?;

        // Load tokenizer and disable built-in padding/truncation.
        // The tokenizer.json may have padding=256 baked in from training;
        // we handle padding ourselves in classify_batch_inner for dynamic padding.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: std::cmp::min(config.max_position_embeddings, 256),
            ..Default::default()
        })).map_err(|e| format!("Truncation config: {e}"))?;

        let max_seq_len = std::cmp::min(config.max_position_embeddings, 256);

        eprintln!(
            "Transformer loaded on {device_name} (hidden={}, layers={}, max_seq={})",
            config.hidden_size, config.num_hidden_layers, max_seq_len
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            max_seq_len,
        })
    }

    fn select_device() -> Device {
        #[cfg(feature = "metal")]
        if let Ok(dev) = Device::new_metal(0) {
            return dev;
        }
        Device::Cpu
    }

    /// Classify a single line. Returns (is_look, p_look).
    pub fn classify_line(&self, line: &str, threshold: f64) -> (bool, f64) {
        match self.classify_batch_inner(&[line], threshold) {
            Ok(results) => results.into_iter().next().unwrap_or((false, 0.0)),
            Err(_) => (false, 0.0),
        }
    }

    /// Classify a batch of lines. Returns Vec<(is_look, p_look)>.
    pub fn classify_batch(
        &self,
        lines: &[&str],
        threshold: f64,
    ) -> Result<Vec<(bool, f64)>, String> {
        self.classify_batch_inner(lines, threshold)
    }

    fn classify_batch_inner(
        &self,
        lines: &[&str],
        threshold: f64,
    ) -> Result<Vec<(bool, f64)>, String> {
        if lines.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = lines.len();

        // Tokenize all lines (tokenizer handles truncation to max_seq_len)
        let encodings = self
            .tokenizer
            .encode_batch(lines.to_vec(), true)
            .map_err(|e| format!("Tokenize: {e}"))?;

        // Dynamic padding: pad to longest sequence in this batch, not max_seq_len.
        // This is a major optimization — most log lines are <64 tokens vs 256 max.
        let pad_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_len))
            .max()
            .unwrap_or(1);

        let mut input_ids_flat = Vec::with_capacity(batch_size * pad_len);
        let mut token_type_ids_flat = Vec::with_capacity(batch_size * pad_len);
        let mut attention_mask_flat = Vec::with_capacity(batch_size * pad_len);

        for enc in &encodings {
            let ids = enc.get_ids();
            let type_ids = enc.get_type_ids();
            let mask = enc.get_attention_mask();
            let len = ids.len().min(pad_len);

            // Copy actual tokens
            input_ids_flat.extend_from_slice(&ids[..len]);
            token_type_ids_flat.extend_from_slice(&type_ids[..len]);
            attention_mask_flat.extend_from_slice(&mask[..len]);

            // Pad remainder
            let pad = pad_len - len;
            if pad > 0 {
                input_ids_flat.extend(std::iter::repeat_n(0u32, pad));
                token_type_ids_flat.extend(std::iter::repeat_n(0u32, pad));
                attention_mask_flat.extend(std::iter::repeat_n(0u32, pad));
            }
        }

        // Build tensors — ensure contiguous layout for Metal kernels
        let input_ids = Tensor::from_vec(input_ids_flat, (batch_size, pad_len), &self.device)
            .and_then(|t| t.contiguous())
            .map_err(|e| format!("Tensor input_ids: {e}"))?;
        let token_type_ids =
            Tensor::from_vec(token_type_ids_flat, (batch_size, pad_len), &self.device)
                .and_then(|t| t.contiguous())
                .map_err(|e| format!("Tensor token_type_ids: {e}"))?;
        let attention_mask =
            Tensor::from_vec(attention_mask_flat, (batch_size, pad_len), &self.device)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.contiguous())
                .map_err(|e| format!("Tensor attention_mask: {e}"))?;

        // Forward pass
        let logits = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| format!("Forward: {e}"))?;

        // Softmax to get probabilities: (batch, 2) where col 0 = P(LOOK), col 1 = P(SKIP)
        let probs = candle_nn::ops::softmax(&logits, 1)
            .map_err(|e| format!("Softmax: {e}"))?;
        let probs_vec: Vec<f32> = probs
            .to_vec2::<f32>()
            .map_err(|e| format!("to_vec2: {e}"))?
            .into_iter()
            .flatten()
            .collect();

        // Extract results
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let p_look = probs_vec[i * 2] as f64;
            let is_look = p_look >= threshold;
            results.push((is_look, p_look));
        }

        Ok(results)
    }

    /// Classify all lines in a file using batched GPU inference.
    pub fn classify_file(
        &self,
        path: &Path,
        threshold: f64,
        max_lines: usize,
        max_look_lines: usize,
        batch_size: usize,
    ) -> Result<TransformerFileResult, String> {
        let file =
            std::fs::File::open(path).map_err(|e| format!("Failed to open file: {e}"))?;
        let reader = BufReader::with_capacity(1024 * 1024, file);

        let start = Instant::now();
        let mut total_lines: usize = 0;
        let mut look_count: usize = 0;
        let mut skip_count: usize = 0;
        let mut look_lines: Vec<(usize, f64, String)> = Vec::new();
        let limit = if max_lines == 0 { usize::MAX } else { max_lines };

        let mut batch_texts: Vec<String> = Vec::with_capacity(batch_size);
        let mut batch_line_nos: Vec<usize> = Vec::with_capacity(batch_size);

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            if total_lines >= limit {
                break;
            }
            total_lines += 1;
            batch_texts.push(line);
            batch_line_nos.push(total_lines);

            if batch_texts.len() >= batch_size {
                self.process_transformer_batch(
                    &batch_texts,
                    &batch_line_nos,
                    threshold,
                    max_look_lines,
                    &mut look_count,
                    &mut skip_count,
                    &mut look_lines,
                )?;
                batch_texts.clear();
                batch_line_nos.clear();
            }
        }

        // Process remaining
        if !batch_texts.is_empty() {
            self.process_transformer_batch(
                &batch_texts,
                &batch_line_nos,
                threshold,
                max_look_lines,
                &mut look_count,
                &mut skip_count,
                &mut look_lines,
            )?;
        }

        let elapsed = start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 {
            total_lines as f64 / elapsed
        } else {
            0.0
        };

        Ok(TransformerFileResult {
            total_lines,
            look_count,
            skip_count,
            look_lines,
            processing_time_s: elapsed,
            lines_per_second: rate,
        })
    }

    fn process_transformer_batch(
        &self,
        texts: &[String],
        line_nos: &[usize],
        threshold: f64,
        max_look_lines: usize,
        look_count: &mut usize,
        skip_count: &mut usize,
        look_lines: &mut Vec<(usize, f64, String)>,
    ) -> Result<(), String> {
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let results = self.classify_batch_inner(&refs, threshold)?;

        for (i, (is_look, p_look)) in results.into_iter().enumerate() {
            if is_look {
                *look_count += 1;
                if look_lines.len() < max_look_lines {
                    look_lines.push((line_nos[i], p_look, texts[i].clone()));
                }
            } else {
                *skip_count += 1;
            }
        }

        Ok(())
    }
}
