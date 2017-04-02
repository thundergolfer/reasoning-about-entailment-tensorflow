WORD2VEC_FILE=word2vec.bin.gz

if [ ! -f $WORD2VEC_FILE ]; then
  echo "Let's download the pre-trained word2vec (it is 1.5GB by the way)"
  if [[ `uname` == "Darwin" ]]; then
    open https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
  else
    xdg-open https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
  fi
fi
