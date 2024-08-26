import unittest
import time
import torch.package


class TestSentimentModel(unittest.TestCase):
    def setUp(self):
        self.model = torch.load(
            "sentiment_model.pt",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False,
        )
        self.model.eval()

    def test_inference_speed(self):
        test_text = "This is a sample text to test inference speed."

        start_time = time.time()
        self.model.infer(test_text)
        end_time = time.time()

        inference_time = end_time - start_time
        self.assertLess(inference_time, 0.1, "Inference should take less than 100ms")

    def test_accuracy(self):
        # You will likely have to change this test to match your own dataset
        test_data = [
            ("This movie was great!", "admiration"),
            ("I hated every minute of it.", "annoyance"),
            ("It was okay, nothing special.", "neutral"),
            ("I wish I had some popcorn.", "desire"),
            ("This movie was terrible!", "disapproval"),
            ("That food was disgusting.", "disgust"),
            ("oh no, I forgot my wallet.", "realization"),
            ("Sadly, my goldfish died recently.", "sadness"),
            ("I'm so excited for the weekend!", "excitement"),
            ("I'm so happy to be here!", "joy"),
            # Add more test cases here
        ]

        correct = 0
        total = len(test_data)

        for text, expected_sentiment in test_data:
            predicted_result = self.model.infer(text)
            predicted_sentiment = predicted_result[0][0]
            print(f"Predicted sentiment: {predicted_sentiment}")
            if predicted_sentiment == expected_sentiment:
                correct += 1

        accuracy = correct / total
        self.assertGreaterEqual(
            accuracy, 0.7, "Model accuracy should be greater than or equal to 70%"
        )


if __name__ == "__main__":
    unittest.main()
