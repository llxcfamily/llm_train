# You shold first build sentencepiece refer to https://github.com/google/sentencepiece
# In the docker, you can run following steps to build sentencepiece
# %git clone https://github.com/google/sentencepiece.git 
# % cd sentencepiece
# % mkdir build
# % cd build
# % cmake ..
# % make -j $(nproc)
# % make install
# % ldconfig -v

# then train spm_model

spm_train --input=processed_data.txt --model_prefix=chinese_sp \
--input_sentence_size=10000000 \
--vocab_size=32000 \
--character_coverage=1.0 \
--model_type=bpe
