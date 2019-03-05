BERT_BASE_DIR=bertbase/uncased_L-12_H-768_A-12
SQUAD_DIR=squad
OUT_DIR=/tmp/BERT_debug_squad

python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/shorttrain.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/shortdev.json \
  --train_batch_size=10 \
  --learning_rate=3e-5 \
  --num_train_epochs=1.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUT_DIR \
  --version_2_with_negative=True

#  --train_file=$SQUAD_DIR/train-v2.0.json \
#  --predict_file=$SQUAD_DIR/dev-v2.0.json \
