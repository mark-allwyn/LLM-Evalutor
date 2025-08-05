"""
Advanced evaluation metrics for LLM outputs.
"""
import re
import string
from typing import Dict, List, Optional, Any
import logging

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
import bert_score

from .exceptions import MetricsError

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Advanced evaluation metrics for LLM outputs."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer model: {e}")
            self.sentence_model = None
    
    def calculate_exact_match(self, reference: str, candidate: str) -> float:
        """Calculate exact match score (0 or 1)."""
        if not reference or not candidate:
            return 0.0
        return 1.0 if reference.strip().lower() == candidate.strip().lower() else 0.0
    
    def calculate_f1_score(self, reference: str, candidate: str) -> float:
        """Calculate F1 score based on token overlap."""
        if not reference or not candidate:
            return 0.0
        
        ref_tokens = set(self._tokenize(reference.lower()))
        cand_tokens = set(self._tokenize(candidate.lower()))
        
        if not ref_tokens and not cand_tokens:
            return 1.0
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        common_tokens = ref_tokens & cand_tokens
        
        if not common_tokens:
            return 0.0
        
        precision = len(common_tokens) / len(cand_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score with smoothing."""
        if not reference or not candidate:
            return 0.0
        
        try:
            reference_tokens = [self._tokenize(reference)]
            candidate_tokens = self._tokenize(candidate)
            
            smoothing_function = SmoothingFunction().method4
            return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        if not reference or not candidate:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bert_score(self, reference: str, candidate: str) -> float:
        """Calculate BERTScore F1."""
        if not reference or not candidate:
            return 0.0
        
        try:
            P, R, F1 = bert_score.score([candidate], [reference], lang='en', verbose=False)
            return F1.item()
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using sentence transformers."""
        if not reference or not candidate or not self.sentence_model:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([reference, candidate])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_length_metrics(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate length-based metrics."""
        ref_tokens = self._tokenize(reference) if reference else []
        cand_tokens = self._tokenize(candidate) if candidate else []
        
        ref_length = len(ref_tokens)
        cand_length = len(cand_tokens)
        
        length_ratio = cand_length / ref_length if ref_length > 0 else (0.0 if cand_length == 0 else float('inf'))
        length_diff = abs(cand_length - ref_length)
        
        return {
            'length_ratio': length_ratio,
            'length_difference': length_diff,
            'candidate_length': cand_length,
            'reference_length': ref_length
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict[str, float]:
        """Calculate readability metrics for the text."""
        if not text:
            return {'avg_sentence_length': 0.0, 'avg_word_length': 0.0}
        
        sentences = self._split_sentences(text)
        words = self._tokenize(text)
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0.0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0.0
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'sentence_count': len(sentences),
            'word_count': len(words)
        }
    
    def calculate_factual_consistency(self, reference: str, candidate: str) -> float:
        """Calculate a simple factual consistency score based on entity overlap."""
        if not reference or not candidate:
            return 0.0
        
        # Simple entity extraction using capitalized words (basic approach)
        ref_entities = set(self._extract_entities(reference))
        cand_entities = set(self._extract_entities(candidate))
        
        if not ref_entities and not cand_entities:
            return 1.0
        if not ref_entities or not cand_entities:
            return 0.0
        
        overlap = len(ref_entities & cand_entities)
        total = len(ref_entities | cand_entities)
        
        return overlap / total if total > 0 else 0.0
    
    def calculate_all_metrics(self, reference: Optional[str], candidate: str) -> Dict[str, float]:
        """Calculate all available metrics."""
        metrics = {
            'exact_match': 0.0,
            'f1_score': 0.0,
            'bleu': 0.0,
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bert_score': 0.0,
            'semantic_similarity': 0.0,
            'length_ratio': 0.0,
            'length_difference': 0.0,
            'factual_consistency': 0.0,
        }
        
        # Add readability metrics for the candidate
        readability = self.calculate_readability_metrics(candidate)
        metrics.update(readability)
        
        # If no reference is provided, only calculate readability metrics
        if reference is None:
            return metrics
        
        # Calculate reference-based metrics
        try:
            metrics['exact_match'] = self.calculate_exact_match(reference, candidate)
            metrics['f1_score'] = self.calculate_f1_score(reference, candidate)
            metrics['bleu'] = self.calculate_bleu(reference, candidate)
            
            rouge_scores = self.calculate_rouge(reference, candidate)
            metrics.update(rouge_scores)
            
            metrics['bert_score'] = self.calculate_bert_score(reference, candidate)
            metrics['semantic_similarity'] = self.calculate_semantic_similarity(reference, candidate)
            
            length_metrics = self.calculate_length_metrics(reference, candidate)
            metrics.update(length_metrics)
            
            metrics['factual_consistency'] = self.calculate_factual_consistency(reference, candidate)
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise MetricsError(f"Failed to calculate metrics: {e}")
        
        return metrics
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split on whitespace
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        return text.lower().split()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Basic sentence splitting on periods, exclamation marks, and question marks
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract simple entities (capitalized words)."""
        # Basic entity extraction - this could be improved with NER
        words = text.split()
        entities = []
        for word in words:
            # Remove punctuation
            clean_word = word.strip(string.punctuation)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
                entities.append(clean_word.lower())
        return entities


class CategorySpecificMetrics:
    """Category-specific evaluation metrics."""
    
    @staticmethod
    def evaluate_math_answer(reference: str, candidate: str) -> Dict[str, float]:
        """Evaluate mathematical answers with tolerance for formatting."""
        if not reference or not candidate:
            return {'math_exact': 0.0, 'math_numeric': 0.0}
        
        # Extract numbers from both answers
        ref_numbers = CategorySpecificMetrics._extract_numbers(reference)
        cand_numbers = CategorySpecificMetrics._extract_numbers(candidate)
        
        exact_match = 1.0 if reference.strip().lower() == candidate.strip().lower() else 0.0
        
        numeric_match = 0.0
        if ref_numbers and cand_numbers:
            # Check if the main numeric answer matches (with small tolerance)
            ref_main = ref_numbers[0] if ref_numbers else None
            cand_main = cand_numbers[0] if cand_numbers else None
            
            if ref_main is not None and cand_main is not None:
                if abs(ref_main - cand_main) < 0.001:  # Small tolerance for floating point
                    numeric_match = 1.0
        
        return {'math_exact': exact_match, 'math_numeric': numeric_match}
    
    @staticmethod
    def evaluate_coding_answer(reference: str, candidate: str) -> Dict[str, float]:
        """Evaluate coding answers."""
        if not reference or not candidate:
            return {'code_similarity': 0.0, 'syntax_valid': 0.0}
        
        # Basic syntax validation (Python-specific)
        syntax_score = CategorySpecificMetrics._check_python_syntax(candidate)
        
        # Code similarity (simplified)
        ref_clean = CategorySpecificMetrics._normalize_code(reference)
        cand_clean = CategorySpecificMetrics._normalize_code(candidate)
        
        # Simple character-based similarity
        similarity = CategorySpecificMetrics._string_similarity(ref_clean, cand_clean)
        
        return {'code_similarity': similarity, 'syntax_valid': syntax_score}
    
    @staticmethod
    def _extract_numbers(text: str) -> List[float]:
        """Extract numbers from text."""
        import re
        # Pattern to match numbers (including decimals, negatives, percentages)
        pattern = r'-?\d+(?:\.\d+)?(?:%)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                if match.endswith('%'):
                    numbers.append(float(match[:-1]))
                else:
                    numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers
    
    @staticmethod
    def _check_python_syntax(code: str) -> float:
        """Check if Python code has valid syntax."""
        try:
            compile(code, '<string>', 'exec')
            return 1.0
        except SyntaxError:
            return 0.0
    
    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize code by removing extra whitespace and comments."""
        lines = code.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            # Strip whitespace
            line = line.strip()
            if line:
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Calculate string similarity using character overlap."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        s1_chars = set(s1.lower())
        s2_chars = set(s2.lower())
        
        intersection = len(s1_chars & s2_chars)
        union = len(s1_chars | s2_chars)
        
        return intersection / union if union > 0 else 0.0
