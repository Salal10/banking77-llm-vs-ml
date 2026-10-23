# Banking77: LLM vs a simple trained model

Is an LLM worth it for sorting bank support messages into 77 categories? I compared Google's Gemini (free tier) with the TF-IDF + logistic regression model from my [support-ticket-triage](https://github.com/Salal10/support-ticket-triage) project, on the same messages.

**Short answer:** not on its own. The simple model was more accurate, about 10,000 times faster, and free. The best setup was a mix: the simple model answers when it's confident, and only the hardest ~5% of messages go to the LLM. On the test set that took accuracy from 90.0% to 91.1%.

## Setup

- Data: [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets), same train/validation split as the earlier project.
- LLM: `gemini-3.5-flash-lite` on the free tier (15 requests/min, 500/day).
- The LLM returns JSON that must be one of the 77 categories, or `out_of_scope` for non-banking messages.
- Development set: 231 validation messages (3 per category), the same for every version.

## Results on the validation sample

| Version | Accuracy | Failed calls | Input tokens per call | Typical latency |
|---|---|---|---|---|
| Trained model (TF-IDF + logistic regression) | 91.3% | 0 | - | under 1 ms |
| LLM, zero-shot (category names only) | 73.6% | 0 | 53 | 1.0 s |
| LLM, few-shot (3 examples per category) | 76.2%* | 14 | 3,915 | 2.9 s |

*14 calls failed with "503: model experiencing high demand" and count as wrong here. On the 217 messages where it did answer, few-shot scored 81.1% vs 72.8% for zero-shot.

What I took from this:

- The LLM didn't know the dataset's labelling habits. Many of its "mistakes" were reasonable. "Want to get a new card in China" is labelled `card_about_to_expire`; the LLM said `country_support`.
- Examples in the prompt helped by about 8 points, but cost 74x more input tokens and made responses about 3x slower.
- The LLM was better with typos and paraphrases. It got "thereto-top" (auto top-up) and "the ATM took my card" right where the trained model didn't.

## Combining the two

The trained model answers first. If its confidence is below a threshold, the LLM answers instead. If the LLM call fails, the trained model's answer is used.

| Threshold | Accuracy | Messages sent to the LLM |
|---|---|---|
| 0.3 | 92.2% | 4.3% |
| 0.4 | 91.8% | 6.5% |
| 0.6 | 90.5% | 19.0% |
| 0.8 | 87.0% | 38.1% |

(Few-shot LLM, validation sample. The trained model alone scored 91.3%.)

Sending more messages to the LLM made things worse. It only helped on the very uncertain ones. The +0.9 points at 0.3 is about 2 messages, so it needed confirming on the test set.

## Final test

I picked threshold 0.3 before looking at the test set, and ran it once.

| System | Test accuracy | Macro F1 | LLM calls |
|---|---|---|---|
| Trained model alone | 90.0% | 0.90 | 0 |
| Trained model + LLM for uncertain messages | 91.1% | 0.90 | 162 of 3,080 (5.3%) |

On just those 162 uncertain messages, the trained model was right 44% of the time and the LLM 65%. So the LLM does help where the simple model struggles. Overall that's about 34 more messages routed correctly, for LLM calls on only 5% of traffic. All 162 calls succeeded.

## Things that went wrong

- **Free-tier limits.** The best Gemini model I could use allowed only 20 requests a day, so I developed on Flash Lite (500 a day). Each model had a different bottleneck: Gemma was limited by tokens per minute, not requests.
- **A dropped connection** stopped a run at message 176 of 231. Results were saved after every call, so re-running carried on from 176. I then added retries for network errors.
- **503 errors made few-shot look worse than zero-shot** at first. Once I counted failures separately, few-shot was clearly better. I now report failed calls as their own column.
- **My token estimate was wrong.** I expected ~850 input tokens per call because of the category list. The measured number was 53.

## Run it

1. Get a free API key from Google AI Studio.
2. Create a `.env` file in the project folder (it's in `.gitignore`):
   ```
   GEMINI_API_KEY=your-key
   ```
3. Install and open the notebooks:
   ```
   uv sync
   uv run jupyter notebook
   ```

Notebooks, in order:

- `01_first_call.ipynb` - first API call, token usage
- `02_structured_classification.ipynb` - JSON output with a fixed list of categories
- `03_eval_v1.ipynb` - zero-shot evaluation vs the trained model
- `04_eval_v2_few_shot.ipynb` - few-shot prompt, comparison, combined setup
- `05_final_test.ipynb` - final test

Shared code is in `src/banking77_llm_vs_ml/core.py`. All results are saved as CSV in `results/`, so you can check the numbers without spending API quota.

## Limitations

- The validation sample is small (231 messages). Differences of 1-2% are noise.
- The threshold was chosen on that same sample.
- The free tier gives no reliability guarantee, as the 503 errors showed.

## How I built it

Built while learning, with an AI assistant as a mentor for concepts and boilerplate. I ran all the experiments, and every number here comes from the files in `results/`.
