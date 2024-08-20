# Sentiment Detection AI
I put this together after the need arose for a simple AI that could detect sentiment. I was unhappy with what I could find so I opted to create my own using 
Google's GoEmotion dataset and modifying it a bit.

## Installation
This was made with python 3.10.11 but might work with other version as well.

Run `pip install -r requirements.txt`.

Run `train_sentiment.py` with your own training values or use the pretrained model from the release tab.

## Usage
```
import torch
import torch.package

# Load model and move to a device
model = torch.package.PackageImporter("sentiment_model.pt").load_pickle("model", "model", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

# Perform inference
print(model.infer("This is a happy little test sentence!"))

```
## Licensing
   Copyright 2023 Foxify52

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.