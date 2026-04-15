# genetic_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os


class VoiceGenotypePredictor:
    """
    Prediction of rs11046212 genotype (ABCC9 gene) from voice features
    Based on GWAS study by Gisladottir et al. 2023
    """

    def __init__(self):
        self.model = None
        self.model_path = "models/genotype_model.pkl"

    def create_synthetic_dataset(self, n_samples=10000):
        """
        Creates a synthetic dataset based on data from the article
        """
        # Genotypes according to Hardy-Weinberg principle (from the article) https://pmc.ncbi.nlm.nih.gov/articles/PMC10256171/#abstract1
        # CC (0 copies of T) — 27.4%, CT (1 copy) — 49.8%, TT (2 copies) — 22.8%
        genotypes = np.random.choice(
            [0, 1, 2],
            n_samples,
            p=[0.274, 0.498, 0.228]
        )

        # Baseline voice parameters
        base_pitch = np.random.normal(145, 25, n_samples) # standard voice pitch

        # Genotype effect: each T copy increases pitch by 0.114 SD (from article)
        # SD pitch ≈ 25 Hz, therefore effect = 0.114 * 25 = 2.85 Hz
        genetic_effect = genotypes * 2.85

        # Final pitch
        final_pitch = base_pitch + genetic_effect

        # Other features correlated with genotype with small genotype effect and have importance 20% in average
        data = pd.DataFrame({
            'genotype': genotypes,
            'pitch_mean': final_pitch,
            'pitch_variability': 0.15 + genetic_effect * 0.002 + np.random.normal(0, 0.03, n_samples),
            'jitter': 0.8 - genetic_effect * 0.01 + np.random.normal(0, 0.15, n_samples),
            'shimmer': 0.35 - genetic_effect * 0.005 + np.random.normal(0, 0.08, n_samples),
            'hnr': 22 + genetic_effect * 0.15 + np.random.normal(0, 3, n_samples),
        })

        return data

    def train(self, n_samples=10000):
        """
        Trains the model on synthetic data
        """
        print(f"Creating synthetic dataset with {n_samples} samples...")
        data = self.create_synthetic_dataset(n_samples)

        X = data[['pitch_mean', 'pitch_variability', 'jitter', 'shimmer', 'hnr']]
        y = data['genotype']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model training
        print("Training Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # Quality assessment
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.1%}")

        # Feature importance
        features = X.columns
        importances = pd.Series(self.model.feature_importances_, index=features)
        print("\nFeature importance:")
        print(importances.sort_values(ascending=False))

        return accuracy

    def save(self):
        """Saves the model to a file"""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        """Loads the model from a file"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("Model loaded")
            return True
        return False

    def predict(self, features_dict):
        """
        Predicts genotype from a feature dictionary

        features_dict must contain:
        - pitch_mean
        - pitch_variability
        - jitter
        - shimmer
        - hnr
        """
        if self.model is None:
            if not self.load():
                print("Model not found, training...")
                self.train()
                self.save()

        # Convert input data to model format
        features = np.array([[
            features_dict.get('pitch_mean', 0),
            features_dict.get('pitch_variability', 0),
            features_dict.get('jitter', 0),
            features_dict.get('shimmer', 0),
            features_dict.get('hnr', 0),
        ]])

        # Prediction and probabilities
        genotype = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Convert numeric genotype to text
        genotype_map = {0: "CC", 1: "CT", 2: "TT"}
        predicted = genotype_map[genotype]

        if predicted == "CC":
            clinical_note = (
                "Your voice pattern suggests the CC genotype (no T allele). "
                "The T allele (a genetic variant in the ABCC9 gene) is associated with higher voice pitch. "
                "Since you don't have T allele, your voice pitch is typically in the lower range. "
            )
        elif predicted == "CT":
            clinical_note = (
                "Your voice pattern suggests the CT genotype (one T allele). "
                "The T allele (a genetic variant in the ABCC9 gene) is associated with higher voice pitch. "
                "With one T allele, your voice pitch may be moderately higher than CC carriers. "
            )
        else:  # TT
            clinical_note = (
                "Your voice pattern suggests the TT genotype (two T alleles). "
                "The T allele (a genetic variant in the ABCC9 gene) is associated with higher voice pitch. "
                "With two T alleles, your voice pitch is typically higher than the population average. "
            )

        return {
            "genotype": predicted,
            "genotype_code": int(genotype),
            "probabilities": {
                "CC": float(probabilities[0]),
                "CT": float(probabilities[1]),
                "TT": float(probabilities[2])
            },
            "snp": "rs11046212",
            "gene": "ABCC9",
            "clinical_note": clinical_note,
            "what_is_T_allele": (
                "What is T allele? "
                "An allele is a variant form of a gene. "
                "The ABCC9 gene has two common variants: C and T. "
                "Studies show that people with T allele tend to have higher voice pitch "
                "(approximately +2.1 Hz per T copy)."
            )
        }


# Create a global instance for use in the application
predictor = VoiceGenotypePredictor()