# Eval3 TOY Video Images

Image-question-answer grounding dataset extracted from the Eval 3 TOY celebrity permutation videos.

Each episode contributes three frames from the first 6 seconds:

- start frame
- middle frame
- end frame

The labels use the user-provided episode block ground truth:

- episodes 0-29: Taylor Swift
- episodes 30-59: Barack Obama
- episodes 60-89: Yann LeCun
- within each 30-episode block: first 10 left, next 10 middle, last 10 right

Files:

- `all.jsonl`: all 270 examples
- `train.jsonl`: 216 examples
- `eval.jsonl`: 54 examples
- `images/`: extracted JPEG frames
- `summary.json`: split metadata

Each JSONL row contains:

```json
{
  "image": "/path/to/image.jpg",
  "episode_index": 0,
  "sample": "start",
  "frame_index": 0,
  "timestamp_s": 0.0,
  "target_person": "Taylor Swift",
  "ground_truth": "left",
  "prompt": "Is Taylor Swift on the left, middle, or right? Answer only one word: left, middle, or right.",
  "answer": "left"
}
```
