
wget -nc https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu
grep -vE "^#" en_ewt-ud-dev.conllu | cut -f 2,4 > data/dev.tt
rm en_ewt-ud-dev.conllu

wget -nc https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
grep -vE "^#" en_ewt-ud-train.conllu | cut -f 2,4 > data/train.tt
rm en_ewt-ud-train.conllu
