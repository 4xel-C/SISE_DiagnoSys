"""
Benchmark Embeddings Module.

This script evaluates different BERT-based models (DrBERT, CamemBERT, BioBERT) 
to determine the best embedding provider for the DiagnoSys RAG pipeline.
"""

import time
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel
from sklearn.metrics import ndcg_score
from deep_translator import GoogleTranslator
import umap
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Models list to benchmark (name: metadata)
# NOTE: Database contains ENGLISH documentation
# French models (DrBERT, CamemBERT) will translate French queries -> English
# English medical models can process directly without translation
MODELS_TO_BENCHMARK = {
    "BioBERT": {
        "path": "dmis-lab/biobert-v1.1",
        "language": "en",
        "description": "Biomedical BERT trained on PubMed + PMC"
    },
    "ClinicalBERT": {
        "path": "emilyalsentzer/Bio_ClinicalBERT",
        "language": "en",
        "description": "Clinical BERT trained on MIMIC-III clinical notes"
    },
    "PubMedBERT": {
        "path": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "language": "en",
        "description": "PubMed BERT trained from scratch on biomedical text"
    },
    "BERT-base": {
        "path": "bert-base-uncased",
        "language": "en",
        "description": "General domain BERT (baseline comparison)"
    },
    "Multi-lingual-E5": {
        "path": "intfloat/multilingual-e5-small",
        "language": "multi",
        "description": "Multilingual model (handles both languages)"
    },
    "Multi-lingual-E5-large": {
        "path": "intfloat/multilingual-e5-large",
        "language": "multi",
        "description": "Multilingual E5 large version (better quality, slower)"
    },
    "BGE-M3": {
        "path": "BAAI/bge-m3",
        "language": "multi",
        "description": "Multilingual BGE model for semantic search"
    },
    "GTE-Multilingual": {
        "path": "Alibaba-NLP/gte-multilingual-base",
        "language": "multi",
        "description": "Alibaba GTE multilingual embedding model"
    },
    "Paraphrase-Multilingual": {
        "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "language": "multi",
        "description": "Sentence-transformers multilingual paraphrase model"
    },
    # "DrBERT": {
    #     "path": "DrBERT/DrBERT-7B",
    #     "language": "fr",
    #     "description": "French medical model (requires FR->EN translation)"
    # },
    # "CamemBERT": {
    #     "path": "almanach/camembert-base",
    #     "language": "fr",
    #     "description": "French general model (requires FR->EN translation)"
    # },
    # "CamemBERT-bio": {
    #     "path": "almanach/camembert-bio-base",
    #     "language": "fr",
    #     "description": "French biomedical CamemBERT (requires FR->EN translation)"
    # },
    "all-MiniLM-L6-v2": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "language": "multi",
        "description": "Lightweight multilingual sentence-transformers model (fast, general purpose)"
    }
}

# Test dataset (Clinical cases vs Reference ENGLISH documents)
# Two query types: 'extracted' (key symptoms) vs 'transcript' (verbose conversation)
TEST_DATA = [
    {
        "query_extracted": "Patient presents productive cough and fever 39°C for 3 days",
        "query_transcript": "Doctor: What brings you in today? Patient: Well, I've been feeling really unwell for the past three days. I have this terrible cough that brings up phlegm, and I've been running a high fever around 39 degrees. Doctor: Any shortness of breath? Patient: Yes, especially when I try to do anything.",
        "expected_doc": "Diagnostic protocol for acute respiratory infections and pneumonia. Clinical presentation typically includes productive cough, fever, and dyspnea.",
        "category": "Pulmonology",
        "query_language": "en"
    },
    {
        "query_extracted": "Chest pain radiating to left arm, sweating, nausea",
        "query_transcript": "Patient: I woke up with this crushing pain in my chest that goes down my left arm. I'm sweating a lot and feel nauseous. Doctor: When did this start? Patient: About an hour ago. It's getting worse. Doctor: On a scale of 1-10? Patient: It's like an 8 or 9.",
        "expected_doc": "Emergency management of Acute Coronary Syndrome (ACS). Typical symptoms include chest pain with radiation, diaphoresis, and nausea.",
        "category": "Cardiology",
        "query_language": "en"
    },
    {
        "query_extracted": "Polyuria, polydipsia, unexplained weight loss",
        "query_transcript": "Patient: I'm constantly thirsty and need to urinate all the time, like every hour. Doctor: How long has this been going on? Patient: Maybe two months. I've also lost about 10 kilos without trying. Doctor: Any family history of diabetes? Patient: Yes, my father has type 2 diabetes.",
        "expected_doc": "Diagnostic criteria and follow-up for type 1 and type 2 diabetes mellitus. Classic triad: polyuria, polydipsia, and weight loss.",
        "category": "Endocrinology",
        "query_language": "en"
    }
]

# Distractor documents for RAG simulation (needle in a haystack test)
# English medical documentation (matching production database)
DISTRACTORS = [
    "Tibial fractures require cast immobilization for 6-8 weeks with regular X-ray follow-up.",
    "Diabetic diet should be low in simple carbohydrates and high in fiber.",
    "Atopic dermatitis is treated with topical corticosteroids and emollients.",
    "Arterial hypertension requires regular blood pressure monitoring and lifestyle modifications.",
    "Influenza vaccination is recommended for elderly patients and immunocompromised individuals.",
    "Appendicitis symptoms include acute abdominal pain in the right lower quadrant.",
    "Asthma treatment includes bronchodilators and inhaled corticosteroids.",
    "Cardiovascular disease prevention includes regular exercise and healthy diet.",
    "Urinary tract infections are more common in women due to anatomical factors.",
    "Breast cancer screening is performed through mammography in women over 50.",
    "Sleep disorders can be treated with benzodiazepines or cognitive behavioral therapy.",
    "Dehydration requires oral or intravenous rehydration depending on severity.",
    "Seasonal allergies are treated with antihistamines and nasal corticosteroids.",
    "Prenatal care includes multiple ultrasound examinations throughout pregnancy.",
    "Stress fractures are common in athletes and require rest and gradual return to activity."
]

class QueryPerturbationGenerator:
    """Generate query variations for robustness testing."""
    
    # Common French medical abbreviations
    ABBREVIATIONS = {
        "tension artérielle": "TA",
        "fréquence cardiaque": "FC",
        "température": "T°",
        "saturation en oxygène": "SpO2",
        "battements par minute": "bpm",
        "antécédents": "ATCD",
        "douleur thoracique": "DT",
        "syndrome coronarien aigu": "SCA",
        "accident vasculaire cérébral": "AVC",
        "insuffisance cardiaque": "IC"
    }
    
    @staticmethod
    def add_typos(text: str, typo_rate: float = 0.1) -> str:
        """Introduce random character typos."""
        words = text.split()
        result = []
        for word in words:
            if random.random() < typo_rate and len(word) > 3:
                pos = random.randint(1, len(word) - 2)
                word = word[:pos] + word[pos+1:]  # Delete character
            result.append(word)
        return " ".join(result)
    
    @staticmethod
    def abbreviate(text: str) -> str:
        """Replace medical terms with abbreviations."""
        for full, abbr in QueryPerturbationGenerator.ABBREVIATIONS.items():
            text = text.replace(full, abbr)
        return text
    
    @staticmethod
    def add_negation(text: str) -> str:
        """Add negation to test semantic understanding."""
        negations = ["pas de", "absence de", "sans"]
        neg = random.choice(negations)
        words = text.split()
        if len(words) > 2:
            pos = random.randint(0, len(words) - 2)
            words.insert(pos, neg)
        return " ".join(words)


class EmbeddingBenchmarker:
    """
    Benchmarker class to evaluate embedding quality and performance.
    
    Attributes:
        models: Dictionary mapping model names to their metadata
        results: List storing benchmark results for each model and test case
        output_dir: Directory path for saving results and visualizations
        translator: Google Translator instance for cross-lingual evaluation
        all_embeddings: Storage for embedding space visualization
    """

    def __init__(self, models: Dict[str, Dict], output_dir: str = "logs"):
        """
        Initialize the benchmark runner.
        
        Args:
            models: Dictionary with model names as keys and metadata dicts as values
            output_dir: Directory to save results (default: "logs")
        """
        self.models = models
        self.results: List[Dict] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.translator = GoogleTranslator(source='fr', target='en')
        self.all_embeddings: Dict[str, List] = {}
        self.perturbation_gen = QueryPerturbationGenerator()

    def _translate_if_needed(self, text: str, model_lang: str, text_lang: str = "en") -> str:
        """
        Translate text if model language doesn't match text language.
        Since documentation is in ENGLISH, French models need French queries translated.
        
        Args:
            text: Original text
            model_lang: Language of the model ('fr', 'en', 'multi')
            text_lang: Language of the input text ('fr' or 'en')
            
        Returns:
            Translated or original text
        """
        # If model is French but text is English, translate EN -> FR
        if model_lang == "fr" and text_lang == "en":
            try:
                translator_en_fr = GoogleTranslator(source='en', target='fr')
                return translator_en_fr.translate(text)
            except Exception as e:
                print(f"    ⚠ EN->FR Translation error: {e}, using original text")
                return text
        
        # If model is English but text is French, translate FR -> EN  
        if model_lang == "en" and text_lang == "fr":
            try:
                translator_fr_en = GoogleTranslator(source='fr', target='en')
                return translator_fr_en.translate(text)
            except Exception as e:
                print(f"    ⚠ FR->EN Translation error: {e}, using original text")
                return text
        
        # Multilingual models or matching languages don't need translation
        return text
    
    def _calculate_advanced_metrics(
        self, 
        query_vec: np.ndarray, 
        corpus_vecs: List[np.ndarray],
        target_index: int,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict:
        """
        Calculate advanced IR metrics (NDCG, Precision@K, Recall@K).
        
        Args:
            query_vec: Query embedding vector
            corpus_vecs: List of document embedding vectors
            target_index: Index of the correct document in corpus
            k_values: List of K values for top-K metrics
            
        Returns:
            Dictionary with metric name-value pairs
        """
        # Calculate similarity scores
        scores = cosine_similarity([query_vec], corpus_vecs)[0]
        ranked_indices = np.argsort(scores)[::-1]
        
        # Create relevance vector (1 for target, 0 for others)
        relevance = np.zeros(len(corpus_vecs))
        relevance[target_index] = 1
        
        # Reorder relevance by ranking
        ranked_relevance = relevance[ranked_indices]
        
        metrics = {}
        
        # NDCG@K
        for k in k_values:
            if k <= len(corpus_vecs):
                # Reshape for ndcg_score
                true_relevance = relevance.reshape(1, -1)
                pred_scores = scores.reshape(1, -1)
                try:
                    ndcg = ndcg_score(true_relevance, pred_scores, k=k)
                    metrics[f"NDCG@{k}"] = round(ndcg, 4)
                except:
                    metrics[f"NDCG@{k}"] = 0.0
        
        # Precision@K and Recall@K
        for k in k_values:
            if k <= len(corpus_vecs):
                top_k_relevant = ranked_relevance[:k].sum()
                precision = top_k_relevant / k
                recall = top_k_relevant / relevance.sum() if relevance.sum() > 0 else 0
                
                metrics[f"Precision@{k}"] = round(precision, 4)
                metrics[f"Recall@{k}"] = round(recall, 4)
                
                # F1@K
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    metrics[f"F1@{k}"] = round(f1, 4)
                else:
                    metrics[f"F1@{k}"] = 0.0
        
        # Mean Average Precision (MAP) - for single query
        rank_of_correct = np.where(ranked_indices == target_index)[0][0] + 1
        metrics["MAP"] = round(1.0 / rank_of_correct, 4)
        
        return metrics

    def load_data(self, filepath: Optional[str] = None) -> List[Dict]:
        """
        Load test data from JSON file or use default TEST_DATA.
        
        Args:
            filepath: Path to JSON file containing test cases (optional)
            
        Returns:
            List of test case dictionaries
        """
        if filepath and Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return TEST_DATA

    def run(
        self, 
        data_file: Optional[str] = None, 
        use_rag_simulation: bool = True,
        test_perturbations: bool = True,
        query_type: str = "both"  # 'extracted', 'transcript', or 'both'
    ):
        """
        Run the benchmark across all defined models.
        
        Args:
            data_file: Optional path to JSON file with test cases
            use_rag_simulation: If True, include RAG simulation with distractors
            test_perturbations: If True, test query variations for robustness
            query_type: Which query format to test - 'extracted' (key symptoms),
                       'transcript' (full conversation), or 'both'
        """
        test_data = self.load_data(data_file)
        
        for name, model_info in self.models.items():
            print(f"\\nTesting model: {name}...")
            print(f"  Language: {model_info['language']} | {model_info['description']}")
            
            try:
                start_init = time.time()
                embeddings_model = HuggingFaceEmbeddings(model_name=model_info["path"])
                init_time = time.time() - start_init
                print(f"  ✓ Initialization: {init_time:.2f}s")

                model_embeddings = []

                for case_idx, case in enumerate(test_data):
                    # Determine query language (default to English for new data)
                    query_lang = case.get("query_language", "en")
                    
                    # Test both query types if requested
                    query_types_to_test = []
                    if query_type in ["extracted", "both"]:
                        query_types_to_test.append("extracted")
                    if query_type in ["transcript", "both"]:
                        query_types_to_test.append("transcript")
                    
                    for q_type in query_types_to_test:
                        # Get appropriate query
                        if q_type == "extracted":
                            query_key = "query_extracted" if "query_extracted" in case else "query"
                        else:
                            query_key = "query_transcript" if "query_transcript" in case else "query"
                        
                        # Skip if query type doesn't exist in case
                        if query_key not in case:
                            continue
                        
                        query_text = case[query_key]
                        
                        # Translate if model language doesn't match query language
                        query = self._translate_if_needed(query_text, model_info["language"], query_lang)
                        expected_doc = self._translate_if_needed(
                            case["expected_doc"], 
                            model_info["language"],
                            "en"  # Docs are always English
                        )
                        
                        start_enc = time.time()
                        
                        # Basic similarity test
                        q_vec = embeddings_model.embed_query(query)
                        d_vec = embeddings_model.embed_query(expected_doc)
                        enc_time = time.time() - start_enc
                        similarity = cosine_similarity([q_vec], [d_vec])[0][0]

                        # Store embeddings for visualization
                        model_embeddings.append({
                            "query": q_vec,
                            "doc": d_vec,
                            "category": case["category"],
                            "type": "query"
                        })

                        result = {
                            "Modèle": name,
                            "Catégorie": case["category"],
                            "Query_ID": case_idx,
                            "Query_Type": q_type,
                            "Similarité": round(similarity, 4),
                            "Temps Encodage (s)": round(enc_time, 4),
                            "Temps Init (s)": round(init_time, 2),
                            "Translated": model_info["language"] != query_lang
                        }
                        
                        # RAG simulation: needle in a haystack
                        if use_rag_simulation:
                            translated_distractors = [
                                self._translate_if_needed(d, model_info["language"], "en") 
                                for d in DISTRACTORS
                            ]
                            rag_metrics = self._run_rag_simulation(
                                embeddings_model, 
                                {"query": query, "expected_doc": expected_doc},
                                translated_distractors
                            )
                            result.update(rag_metrics)
                        
                        self.results.append(result)
                        
                        # Test query perturbations for robustness (only for extracted queries)
                        if test_perturbations and q_type == "extracted" and query_lang == "fr":
                            # Only perturb French queries
                            self._test_query_perturbations(
                                embeddings_model, 
                                {**case, "query": query_text}, 
                                name, 
                                case_idx
                            )
                
                # Store embeddings for this model
                self.all_embeddings[name] = model_embeddings
                    
                print(f"  ✓ Processed {len(test_data)} test cases")
                
            except Exception as e:
                print(f"  ✗ Error testing {name}: {e}")

    def _test_query_perturbations(
        self,
        model: HuggingFaceEmbeddings,
        case: Dict,
        model_name: str,
        case_idx: int
    ) -> None:
        """Test model robustness against query variations."""
        original_query = case["query"]
        expected_doc = case["expected_doc"]
        
        # Get original embedding
        original_vec = model.embed_query(original_query)
        doc_vec = model.embed_query(expected_doc)
        original_sim = cosine_similarity([original_vec], [doc_vec])[0][0]
        
        perturbations = {
            "typos": self.perturbation_gen.add_typos(original_query),
            "abbreviations": self.perturbation_gen.abbreviate(original_query),
        }
        
        for pert_type, perturbed_query in perturbations.items():
            if perturbed_query != original_query:  # Only if actually changed
                pert_vec = model.embed_query(perturbed_query)
                pert_sim = cosine_similarity([pert_vec], [doc_vec])[0][0]
                
                # Calculate robustness (similarity degradation)
                degradation = original_sim - pert_sim
                
                self.results.append({
                    "Modèle": model_name,
                    "Catégorie": case["category"],
                    "Query_ID": case_idx,
                    "Similarité": round(pert_sim, 4),
                    "Temps Encodage (s)": 0.0,
                    "Temps Init (s)": 0.0,
                    "Translated": False,
                    "Perturbation": pert_type,
                    "Degradation": round(degradation, 4)
                })

    def _run_rag_simulation(
        self, 
        model: HuggingFaceEmbeddings, 
        case: Dict, 
        distractors: List[str]
    ) -> Dict:
        """
        Simulate RAG retrieval by mixing the correct document with distractors.
        
        Args:
            model: Embedding model to test
            case: Test case dictionary with query and expected_doc
            distractors: List of distractor documents
            
        Returns:
            Dictionary with RAG metrics (MRR, position, advanced metrics)
        """
        # Create corpus: target document + distractors
        corpus = distractors + [case["expected_doc"]]
        
        # Encode all documents
        corpus_vecs = model.embed_documents(corpus)
        query_vec = model.embed_query(case["query"])
        
        # Calculate similarity scores
        scores = cosine_similarity([query_vec], corpus_vecs)[0]
        
        # Find rank of the correct document
        ranked_indices = np.argsort(scores)[::-1]
        target_index = len(distractors)  # Target is last in corpus
        rank = np.where(ranked_indices == target_index)[0][0] + 1
        
        # Calculate MRR (Mean Reciprocal Rank)
        mrr = 1.0 / rank
        
        # Calculate advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(
            query_vec, corpus_vecs, target_index
        )
        
        result = {
            "MRR": round(mrr, 4),
            "Position": rank,
            "Top-1": 1 if rank == 1 else 0,
            "Top-3": 1 if rank <= 3 else 0,
            "Top-5": 1 if rank <= 5 else 0,
            "Top-10": 1 if rank <= 10 else 0
        }
        
        result.update(advanced_metrics)
        return result

    def calculate_statistical_significance(self) -> pd.DataFrame:
        """
        Perform paired t-tests between models to determine statistical significance.
        
        Returns:
            DataFrame with p-values and effect sizes for model comparisons
        """
        df = pd.DataFrame(self.results)
        
        # Filter out perturbation results for fair comparison
        df_clean = df[~df.get("Perturbation", pd.Series([None]*len(df))).notna()]
        
        if len(df_clean) == 0 or "MRR" not in df_clean.columns:
            print("Not enough data for statistical testing")
            return pd.DataFrame()
        
        models = df_clean["Modèle"].unique()
        comparisons = []
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Get MRR scores for both models
                scores1 = df_clean[df_clean["Modèle"] == model1]["MRR"].values
                scores2 = df_clean[df_clean["Modèle"] == model2]["MRR"].values
                
                if len(scores1) == len(scores2) and len(scores1) > 1:
                    # Paired t-test
                    t_stat, p_value = ttest_rel(scores1, scores2)
                    
                    # Cohen's d (effect size)
                    diff = scores1 - scores2
                    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
                    
                    # 95% Confidence interval
                    ci_95 = stats.t.interval(
                        0.95, len(diff)-1, 
                        loc=np.mean(diff), 
                        scale=stats.sem(diff)
                    )
                    
                    comparisons.append({
                        "Model_A": model1,
                        "Model_B": model2,
                        "Mean_MRR_A": round(np.mean(scores1), 4),
                        "Mean_MRR_B": round(np.mean(scores2), 4),
                        "p_value": round(p_value, 4),
                        "significant": "Yes" if p_value < 0.05 else "No",
                        "Cohen_d": round(cohens_d, 4),
                        "CI_95_lower": round(ci_95[0], 4),
                        "CI_95_upper": round(ci_95[1], 4)
                    })
        
        return pd.DataFrame(comparisons)
    
    def analyze_failures(self) -> Dict:
        """
        Analyze failure patterns across models and categories.
        
        Returns:
            Dictionary with failure analysis insights
        """
        df = pd.DataFrame(self.results)
        
        # Filter clean results (no perturbations)
        df_clean = df[~df.get("Perturbation", pd.Series([None]*len(df))).notna()]
        
        if len(df_clean) == 0 or "Position" not in df_clean.columns:
            return {}
        
        analysis = {
            "worst_categories": {},
            "best_categories": {},
            "failure_rate_by_model": {}
        }
        
        # Analyze by category
        for category in df_clean["Catégorie"].unique():
            cat_data = df_clean[df_clean["Catégorie"] == category]
            avg_position = cat_data["Position"].mean()
            analysis["worst_categories"][category] = round(avg_position, 2)
        
        # Sort categories
        sorted_worst = dict(sorted(
            analysis["worst_categories"].items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        analysis["worst_categories"] = dict(list(sorted_worst.items())[:3])
        
        sorted_best = dict(sorted(
            analysis["worst_categories"].items(), 
            key=lambda x: x[1]
        ))
        analysis["best_categories"] = dict(list(sorted_best.items())[:3])
        
        # Failure rate by model (failure = not in top-3)
        for model in df_clean["Modèle"].unique():
            model_data = df_clean[df_clean["Modèle"] == model]
            if "Top-3" in model_data.columns:
                failure_rate = 1 - model_data["Top-3"].mean()
                analysis["failure_rate_by_model"][model] = round(failure_rate, 4)
        
        return analysis
    
    def plot_results(self) -> None:
        """
        Generate visualization plots for benchmark results.
        Saves comprehensive graphs with advanced metrics.
        """
        df = pd.DataFrame(self.results)
        
        if df.empty:
            print("No results to plot.")
            return
        
        # Filter out perturbation results for main plots
        df_clean = df[~df.get("Perturbation", pd.Series([None]*len(df))).notna()]
        
        # Configure style
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(3, 2, figsize=(18, 20))
        
        # Plot 1: Semantic Quality (Cosine Similarity)
        if "Similarité" in df_clean.columns and "Catégorie" in df_clean.columns:
            sns.barplot(
                x="Modèle", y="Similarité", hue="Catégorie", 
                data=df_clean, ax=axes[0, 0], palette="viridis"
            )
            axes[0, 0].set_title("Qualité Sémantique (Cosine Similarity)", fontsize=14, fontweight='bold')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].legend(title="Catégorie", loc="lower right", fontsize=8)
        
        # Plot 2: Encoding Speed
        if "Temps Encodage (s)" in df_clean.columns:
            avg_time = df_clean.groupby("Modèle")["Temps Encodage (s)"].mean().reset_index()
            sns.barplot(
                x="Modèle", y="Temps Encodage (s)", 
                data=avg_time, ax=axes[0, 1], palette="magma"
            )
            axes[0, 1].set_title("Latence Moyenne d'Encodage", fontsize=14, fontweight='bold')
        
        # Plot 3: RAG Performance (MRR)
        if "MRR" in df_clean.columns:
            avg_mrr = df_clean.groupby("Modèle")["MRR"].mean().reset_index()
            sns.barplot(
                x="Modèle", y="MRR", 
                data=avg_mrr, ax=axes[1, 0], palette="rocket"
            )
            axes[1, 0].set_title("Performance RAG (Mean Reciprocal Rank)", fontsize=14, fontweight='bold')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Fair threshold')
        
        # Plot 4: Top-K Accuracy
        if "Top-1" in df_clean.columns:
            top_k_cols = [col for col in ["Top-1", "Top-3", "Top-5", "Top-10"] if col in df_clean.columns]
            top_k_data = df_clean.groupby("Modèle")[top_k_cols].mean().reset_index()
            top_k_melted = top_k_data.melt(id_vars="Modèle", var_name="Metric", value_name="Accuracy")
            sns.barplot(
                x="Modèle", y="Accuracy", hue="Metric",
                data=top_k_melted, ax=axes[1, 1], palette="mako"
            )
            axes[1, 1].set_title("Top-K Accuracy", fontsize=14, fontweight='bold')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].legend(title="Metric", fontsize=8)
        
        # Plot 5: NDCG@K Comparison
        if "NDCG@3" in df_clean.columns:
            ndcg_cols = [col for col in df_clean.columns if col.startswith("NDCG@")]
            if ndcg_cols:
                ndcg_data = df_clean.groupby("Modèle")[ndcg_cols].mean().reset_index()
                ndcg_melted = ndcg_data.melt(id_vars="Modèle", var_name="Metric", value_name="NDCG")
                sns.barplot(
                    x="Modèle", y="NDCG", hue="Metric",
                    data=ndcg_melted, ax=axes[2, 0], palette="crest"
                )
                axes[2, 0].set_title("NDCG@K (Normalized Discounted Cumulative Gain)", fontsize=14, fontweight='bold')
                axes[2, 0].set_ylim(0, 1)
                axes[2, 0].legend(title="K", fontsize=8)
        
        # Plot 6: Robustness (Perturbation Impact)
        if "Perturbation" in df.columns:
            df_pert = df[df["Perturbation"].notna()]
            if not df_pert.empty and "Degradation" in df_pert.columns:
                pert_avg = df_pert.groupby(["Modèle", "Perturbation"])["Degradation"].mean().reset_index()
                sns.barplot(
                    x="Modèle", y="Degradation", hue="Perturbation",
                    data=pert_avg, ax=axes[2, 1], palette="flare"
                )
                axes[2, 1].set_title("Robustness (Lower = Better)", fontsize=14, fontweight='bold')
                axes[2, 1].legend(title="Perturbation Type", fontsize=8)
                axes[2, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / "benchmark_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\\n✓ Graphs saved to {output_path}")
        plt.close()
    
    def plot_embedding_space(self) -> None:
        """Visualize embedding space using UMAP dimensionality reduction."""
        if not self.all_embeddings:
            print("No embeddings to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (model_name, embeddings) in enumerate(self.all_embeddings.items()):
            if idx >= 4:
                break
            
            # Extract query vectors and categories
            vectors = [e["query"] for e in embeddings]
            categories = [e["category"] for e in embeddings]
            
            if len(vectors) < 2:
                continue
            
            # Apply UMAP
            reducer = umap.UMAP(n_neighbors=min(5, len(vectors)-1), random_state=42)
            embedding_2d = reducer.fit_transform(np.array(vectors))
            
            # Plot
            unique_cats = list(set(categories))
            colors = sns.color_palette("husl", len(unique_cats))
            
            for cat, color in zip(unique_cats, colors):
                mask = np.array([c == cat for c in categories])
                axes[idx].scatter(
                    embedding_2d[mask, 0], 
                    embedding_2d[mask, 1],
                    c=[color], label=cat, alpha=0.7, s=100
                )
            
            axes[idx].set_title(f"{model_name} - Embedding Space", fontsize=12, fontweight='bold')
            axes[idx].legend(loc="best", fontsize=8)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "embedding_space_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Embedding space visualization saved to {output_path}")
        plt.close()
        """
        Generate visualization plots for benchmark results.
        Saves two graphs: semantic quality and encoding speed.
        """
        df = pd.DataFrame(self.results)
        
        if df.empty:
            print("No results to plot.")
            return
        
        # Configure style
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Semantic Quality (Cosine Similarity)
        sns.barplot(
            x="Modèle", y="Similarité", hue="Catégorie", 
            data=df, ax=axes[0, 0], palette="viridis"
        )
        axes[0, 0].set_title("Qualité Sémantique (Cosine Similarity)", fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend(title="Catégorie", loc="lower right")
        
        # Plot 2: Encoding Speed
        avg_time = df.groupby("Modèle")["Temps Encodage (s)"].mean().reset_index()
        sns.barplot(
            x="Modèle", y="Temps Encodage (s)", 
            data=avg_time, ax=axes[0, 1], palette="magma"
        )
        axes[0, 1].set_title("Latence Moyenne d'Encodage", fontsize=14, fontweight='bold')
        
        # Plot 3: RAG Performance (MRR)
        if "MRR" in df.columns:
            avg_mrr = df.groupby("Modèle")["MRR"].mean().reset_index()
            sns.barplot(
                x="Modèle", y="MRR", 
                data=avg_mrr, ax=axes[1, 0], palette="rocket"
            )
            axes[1, 0].set_title("Performance RAG (Mean Reciprocal Rank)", fontsize=14, fontweight='bold')
            axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Top-k Accuracy
        if "Top-1" in df.columns:
            top_k_data = df.groupby("Modèle")[["Top-1", "Top-3"]].mean().reset_index()
            top_k_melted = top_k_data.melt(id_vars="Modèle", var_name="Metric", value_name="Accuracy")
            sns.barplot(
                x="Modèle", y="Accuracy", hue="Metric",
                data=top_k_melted, ax=axes[1, 1], palette="mako"
            )
            axes[1, 1].set_title("Top-K Accuracy", fontsize=14, fontweight='bold')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        output_path = self.output_dir / "benchmark_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphs saved to {output_path}")
        plt.close()

    def export_results(self, skip_umap: bool = False) -> None:
        """
        Export results to CSV and display comprehensive summary statistics.
        
        Args:
            skip_umap: If True, skip UMAP visualization to save time
        """
        df = pd.DataFrame(self.results)
        
        if df.empty:
            print("No results to export.")
            return
        
        # Filter clean vs perturbation results
        df_clean = df[~df.get("Perturbation", pd.Series([None]*len(df))).notna()]
        
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        # Basic metrics
        summary_cols = ["Similarité", "Temps Encodage (s)"]
        if "MRR" in df_clean.columns:
            summary_cols.extend(["MRR", "Top-1", "Top-3", "Top-5"])
            if "NDCG@3" in df_clean.columns:
                summary_cols.extend(["NDCG@3", "NDCG@5", "MAP"])
        
        summary = df_clean.groupby("Modèle")[summary_cols].mean()
        print("\n" + summary.to_string())
        
        # Statistical significance testing
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTING (Paired t-tests)")
        print("="*70)
        sig_results = self.calculate_statistical_significance()
        if not sig_results.empty:
            print("\n" + sig_results.to_string(index=False))
            
            # Save statistical results
            sig_path = self.output_dir / "statistical_significance.csv"
            sig_results.to_csv(sig_path, index=False)
            print(f"\n✓ Statistical tests saved to {sig_path}")
        
        # Failure analysis
        print("\n" + "="*70)
        print("FAILURE ANALYSIS")
        print("="*70)
        failure_analysis = self.analyze_failures()
        if failure_analysis:
            print("\nWorst performing categories (avg rank):")
            for cat, rank in failure_analysis.get("worst_categories", {}).items():
                print(f"  - {cat}: {rank}")
            
            print("\nFailure rate by model (% not in Top-3):")
            for model, rate in failure_analysis.get("failure_rate_by_model", {}).items():
                print(f"  - {model}: {rate*100:.1f}%")
        
        # Perturbation results
        df_pert = df[df.get("Perturbation", pd.Series([None]*len(df))).notna()]
        if not df_pert.empty:
            print("\n" + "="*70)
            print("ROBUSTNESS TESTING (Query Perturbations)")
            print("="*70)
            pert_summary = df_pert.groupby(["Modèle", "Perturbation"])["Degradation"].mean()
            print("\nAverage similarity degradation:")
            print(pert_summary.to_string())
        
        # Export to CSV
        csv_path = self.output_dir / "embedding_benchmark_detailed.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Detailed results saved to {csv_path}")
        
        # Generate visualizations
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        self.plot_results()
        
        if not skip_umap:
            self.plot_embedding_space()
        else:
            print("⚡ Skipping UMAP visualization (faster benchmark)")

if __name__ == "__main__":
    print("="*70)
    print("EMBEDDING MODELS BENCHMARK FOR RAG PIPELINE - ENHANCED VERSION")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Cross-lingual evaluation (EN models with translation)")
    print("  ✓ Advanced IR metrics (NDCG@K, Precision@K, Recall@K, MAP)")
    print("  ✓ Query perturbation testing (typos, abbreviations)")
    print("  ✓ Statistical significance testing (paired t-tests)")
    print("  ✓ Embedding space visualization (UMAP)")
    print("  ✓ Comprehensive failure analysis\n")
    
    benchmarker = EmbeddingBenchmarker(MODELS_TO_BENCHMARK)
    
    # Run benchmark with all enhancements
    # Options:
    # - data_file="data/comprehensive_benchmark.json" for extended dataset
    # - use_rag_simulation=True for needle-in-haystack testing
    # - test_perturbations=True for robustness evaluation
    # QUICK BENCHMARK (faster, less comprehensive)
    # Uncomment to test only key models on extracted queries:
    # benchmarker.run(
    #     data_file="data/comprehensive_benchmark_en.json",
    #     use_rag_simulation=True,
    #     test_perturbations=False,  # Skip perturbations (saves ~3 min)
    #     query_type="extracted"  # Only test extracted (saves 50% time)
    # )
    
    # FULL BENCHMARK (comprehensive but slower)
    benchmarker.run(
        data_file="data/comprehensive_benchmark_en.json",
        use_rag_simulation=True,
        test_perturbations=True,
        query_type="both"  # Test both extracted and transcript queries
    )
    
    # Export results and generate visualizations
    benchmarker.export_results(skip_umap=True)  # Skip UMAP to save time
    
    print("\n" + "="*70)
    print("✓ BENCHMARK COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated files:")
    print("  - logs/embedding_benchmark_detailed.csv")
    print("  - logs/statistical_significance.csv")
    print("  - logs/benchmark_results.png")
    print("  - logs/embedding_space_visualization.png")