spm_train --input=processed_data.txt --model_prefix=spm.model \
--input_sentence_size=20000000 \
--seed_sentencepiece_size=20000000 \
--max_sentence_length=41920  \
--vocab_size=32000 \
--character_coverage=0.9995 \
--model_type=bpe

