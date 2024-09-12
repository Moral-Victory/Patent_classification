# Patent_classification

## Can we do better with transformers?

### 🤗 DistilBERT

The DistilBERT model was first proposed in the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). It has **40%** less parameters than `bert-base-uncased`, runs **60%** faster while preserving over **95%** of BERT’s performance as measured on the GLUE language understanding benchmark.

#### Data preprocessing and tokenization

We use the `distilbert-base-uncased` tokenizer. Case-sensitivity is not a concern in this dataset because typical patents we encounter consist of well-formatted text with almost no typos/misspellings, and we would expect words in the data to retain context regardless of capitalization.

The data is loaded and transformed (i.e., encoded into input IDs with attention masks) through a combination of the Hugging Face [Datasets library](https://huggingface.co/docs/datasets/), as well as their [Tokenizers library](https://github.com/huggingface/tokenizers). The Datasets pipeline allows us to easily generate train/validation/test splits from a range of raw data sources, and the Tokenizers pipeline efficiently encodes the vocabulary of the dataset into a form that the DistilBERT `trainer` instance can make use of.

#### Model training
The model is trained using the `classifier_distilbert_train.py` script provided in this repo as follows.

```
$ python3 classifier_distilbert_train.py
```
Verify that the training loss goes down in each epoch, and that the validation F1 increases accordingly. This outputs the model weights to the `pytorch_model/` directory

#### Model optimization and compression

A big concern with deep learning models is the computational cost associated with making inferences on real world data in production. One approach to make the inference process more efficient is to optimize and quantize the PyTorch model via [ONNX](https://github.com/onnx/onnx), an open source framework that provides a standard interface for optimizing deep learning models and their computational graphs.

On average, a **10x-30x** speedup in CPU-based inference, along with a **4x** reduction in model size is possible for an optimized, quantized DistilBERT-ONNX model (compared to the base DistilBERT-PyTorch model that we trained on GPU).

### Use 🤗 Hugging Face command line module to convert to ONNX

See the [PyTorch documentation](https://pytorch.org/docs/stable/quantization.html) for a more detailed description of quantization, as well as the difference between static and dynamic quantization.

The following command is used to convert the PyTorch model to an ONNX model. First, `cd` to an **empty directory** in which we want the ONNX model file to be saved, and then specify the source PyTorch model path (that contains a valid `config.json`) in relation to the current path. An example is shown below.

```sh
# Assuming the PyTorch model weights (config.json and 
# pytorch_model.bin file) are in the pytorch_model/ directory
$ cd onnx_model
$ python3 -m transformers.convert_graph_to_onnx \
  --framework pt \
  --model pytorch_model \
  --tokenizer distilbert-base-uncased \
  --quantize onnx_model \
  --pipeline sentiment-analysis
```

Note that we need to specify the `--pipeline sentiment-analysis` argument to avoid input array broadcasting issues as per the Hugging Face API. Specifying the `sentiment-analysis` argument forces it to use sequence classification tensor shapes during export, so the correct outputs are sent to the ONNX compute layers.

The quantized ONNX model file is then generated with in the current directory, which can then be used to make much more rapid inferences on CPU.

### DistilBERT results
The evaluation script `classifier_distilbert_evaluate.py` is run to produce the following results.

```
$ python3 classifier_distilbert_evaluate.py
```

```
Macro F1: 90.687 %
Micro F1: 91.027 %
Weighted F1: 91.033 %
Accuracy: 91.027 %
```

![](img/distilbert_results_normalized.png)

The confusion matrix shows that the DistilBERT model's results are much, much better than the baseline model's. This makes sense because the pretrained transformer + a better training regime during fine-tuning (including a warmup of the learning rate and more robust optimization) helps the model better disambiguate tokens from much smaller amounts of training data.

However, even though we see a 100% prediction rate for the minority class 'D', the un-normalized confusion matrix (on the right of the image above) shows that we only made predictions on only **4** test samples for this class, as can be seen below.

![](img/distilbert_results_unnormalized.png)

Thus, it is a bit premature to state that the DistilBERT classifier is *truly* performing well, with such a limited test sample size on certain classes. To gain a better understanding of how this DistilBERT classifier will actually perform in the wild, it would make sense to scrape a random set of around 100 samples from the minority classes ('D' and 'E') from a much larger time period, and seeing what percentage of those are predicted correctly.

However, even without cost-sensitive weights in this case, and with such an imbalanced dataset, it's encouraging that the DistilBERT classifier is showing such good results!

#### Note on cost-sensitive learning for transformers

Just like in the case with the SVM, it is possible to perform cost-sensitive weighting for the transformer model by subclassing the `Trainer` instance and passing the class weights to the `CrossEntropy` loss as follows:

```py
class CostSensitiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss
```

See [this GitHub issue](https://github.com/huggingface/transformers/issues/7024) on the 🤗 Hugging Face transformers repo for more details.

Running this trainer instance could potentially help the model generalize better to unseen vocabulary, although initial results show that it might not be necessary.

#### Additional experiments with transformers
A better way to study the transformer model's real-world performance would be to look at the effect of more balanced training data on classification performance from a much larger sample of unseen data. This can be done by scraping and obtaining more patent data over multiple months for the minority classes ("D" and "E"), so that the model sees a larger vocabulary over a longer time period, allowing it to generalize better.
