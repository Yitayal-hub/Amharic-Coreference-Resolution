Amharic Coreference Resolution with Integrated NER and Multi-Head Attention
   It is an advanced coreference resolution system that integrates Named Entity Recognition (NER) with multi-head attention mechanisms. 


#Installation

Prerequisites

Python 3.6+

TensorFlow 2.0+

Other dependencies listed in requirements.txt

#Setup

Install dependencies:

bash
pip3 install -r requirements.txt

Build custom TensorFlow kernels:

bash -x -e setup_all.sh

Prepare training data and download Amharic word2vec from this link https://sparknlp.org/2022/03/14/w2v_cc_300d_am_3_0.html:
 Divide the dataset into 80%, 20%, and 20% for training, development and testing data respectively.

Prepare Embeddings:
 bash -x -e setup_training.sh

    # Filter embeddings for your dataset
    python3 filter_embeddings.py cc.am.300.vec train.jsonlines dev.jsonlines test.jsonlines

    # Generate character vocabulary
    python3 get_char_vocab.py


#Usage

Training

bash
python3 train.py <experiment_name> [--logdir <log_directory>]

Example:

bash
python3 train.py train_am_coref --logdir logs/amharic_coref

Evaluation
bash
python3 evaluate.py <experiment_name> <eval_data_path> [<output_file>]

Examples:

bash
# Single model evaluation
python3 evaluate.py train_am_mentcoref test.amharic.jsonlines predictions.jsonlines

# Two-model pipeline
python3 evaluate.py train_am_ment,train_am_coref test.amharic.jsonlines final_predictions.jsonlines


Output Format

The system outputs predictions in JSON Lines format with the following structure:

json
{
  "doc_key": "document_id",
  "clusters": [[[start, end], ...], ...],
  "sentences": [...],
  "predicted_ner": ["PER", "O", "ORG", ...],
  ...
}
