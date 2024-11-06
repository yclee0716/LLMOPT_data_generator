from nlp_parser import NLPParser
from feature_extractor import FeatureExtractor
from scoring import Scoring

class DifficultyClassifier:
    def __init__(self):
        self.parser = NLPParser()
        self.extractor = FeatureExtractor()
        self.scorer = Scoring()

    def classify_difficulty(self, problem_text):
        # Step 1: Parse the text
        parsed_data = self.parser.parse(problem_text)
        
        # Step 2: Extract features
        features = self.extractor.extract_features(parsed_data)
        
        # Step 3: Score the problem
        difficulty_score = self.scorer.score_problem(features)
        
        return difficulty_score

if __name__ == "__main__":
    classifier = DifficultyClassifier()
    problem_text = "Solve an integer knapsack problem with 10 variables, maximizing total value subject to weight constraints."
    score = classifier.classify_difficulty(problem_text)
    print("The difficulty score of the given problem is:", score)
