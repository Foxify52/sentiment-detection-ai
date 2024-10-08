# Sentiment Detection AI
I put this together after the need arose for a simple AI that could detect sentiment. I was unhappy with what I could find so I opted to create my own using a modified version of Google's GoEmotion dataset.

## Installation
This was made with python 3.10.11 but might work with other version as well.

Run `poetry install --no-root`.

Run `train_sentiment.py` with your own hyperparameters or use the pretrained model from the release tab.

Recommended to run `test_sentiment.py` to test after training with your own data or hyperparameters.

## Usage
```
import torch

# Load model and move to a device
model = torch.load(
            "sentiment_model.pt",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False,
        )
model.eval()

# Perform inference
print(model.infer("This is a happy little test sentence!"))

```
## Licensing
   Copyright 2024 Foxify52

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.