# How I got my high BLUE score

First of: I've got this score probably due to the way the data was configured. 
I'm currently rerunning everything so we can be sure it was the data and not because of a change from SRILM to KENLM.

Regardless, let me now explain the steps I took.

While debugging moses, the main problem was getting the tuning process to start. Due to this, I split the process.sh file
into two parts: process-train.sh and process-tune.sh

Process-train.sh contains the following lines from the bash script:
```
#!/bin/bash
set -e

WD=`pwd`
# Replace the following path by your own path
MT_SCRIPTS=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/scripts/tokenizer
SCRIPTS_ROOTDIR=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/scripts
MOSES_DIR=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder
# the path to Moses
MOSES=`which moses` \
        && type -P $MOSES &>/dev/null \
            || { echo "Could not find Moses at '$MOSES'. Aborting." >&2; exit 1; }
# the path to ngram-count of SRILM
NGRAM_COUNT=`which ngram-count` \
        && type -P $NGRAM_COUNT &>/dev/null \
            || { echo "Could not find SRILM at '$NGRAM_COUNT'. Aborting." >&2; exit 1; }
type -P $MT_SCRIPTS/tokenizer.perl &>/dev/null || { echo "Could not find Tokenizer at $MT_SCRIPTS/tokenizer.perl. Aborting." >&2; exit 1; }
echo "Lowercasing"
[[ -e data/low ]] && rm -r data/low; mkdir data/low
for f in data/tok/*; do
    echo " $f"
    $MT_SCRIPTS/lowercase.perl < $f > data/low/`basename $f`;
done

# 4) train the language models
[[ -e lm ]] && rm -r lm; mkdir lm
for f in en ja; do
    echo "Training $f LM"
    $NGRAM_COUNT -order 5 -interpolate -kndiscount -text data/low/kyoto-train.cln.$f -lm lm/kyoto-train.$f
done

# 5) train moses
for f in ja en; do
    if [[ $f = "ja" ]]; then e="en"; else e="ja"; fi
    [[ -e $f-$e ]] && rm -r $f-$e; mkdir $f-$e
    echo "Training Moses for $f-$e"
$SCRIPTS_ROOTDIR/training/train-model.perl  -root-dir $f-$e -corpus data/low/kyoto-train.cln -f $f -e $e -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:5:$WD/lm/kyoto-train.$e:0 -external-bin-dir $MOSES_DIR/tools
done
```
process-tune.sh contains the second half of the original process.sh script + the needed paths:
```
#!/bin/bash
set -e

WD=`pwd`
# Replace the following path by your own path
MT_SCRIPTS=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/scripts/tokenizer
SCRIPTS_ROOTDIR=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/scripts
MOSES_DIR=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder
# the path to Moses
MOSES=`which moses` \
        && type -P $MOSES &>/dev/null \
            || { echo "Could not find Moses at '$MOSES'. Aborting." >&2; exit 1; }
# the path to ngram-count of SRILM
NGRAM_COUNT=`which ngram-count` \
        && type -P $NGRAM_COUNT &>/dev/null \
            || { echo "Could not find SRILM at '$NGRAM_COUNT'. Aborting." >&2; exit 1; }
type -P $MT_SCRIPTS/tokenizer.perl &>/dev/null || { echo "Could not find Tokenizer at $MT_SCRIPTS/tokenizer.perl. Aborting." >&2; exit 1; }

# 6) do mert
for f in ja en; do
    if [[ $f = "ja" ]]; then e="en"; else e="ja"; fi
    echo "Doing MERT for $f-$e"
    $SCRIPTS_ROOTDIR/training/mert-moses.pl $WD/data/low/kyoto-tune.$f $WD/data/low/kyoto-tune.$e $MOSES $WD/$f-$e/model/moses.ini --working-dir $WD/$f-$e/tuning --mertdir $MOSES_DIR/bin  --pairwise-ranked --decoder-flags="-threads 30 -v 0"
    #$SCRIPTS_ROOTDIR/training/mert-moses.pl $WD/data/low/kyoto-tune.$f $WD/data/low/kyoto-tune.$e $MOSES $WD/$f-$e/model/moses.ini --working-dir $WD/$f-$e/tuning --mertdir $MOSES_DIR/bin --decoder-flags="-threads 20"
done

# 7) test
for f in ja en; do
    if [[ $f = "ja" ]]; then e="en"; else e="ja"; fi
    [[ -e $f-$e/evaluation ]] || mkdir $f-$e/evaluation
    for t in dev test; do
        echo "Testing and evaluating for $f-$e, $t"
        # filter
        echo "$SCRIPTS_ROOTDIR/training/filter-model-given-input.pl $f-$e/evaluation/filtered.$t $f-$e/tuning/moses.ini data/low/kyoto-$t.$f"
        $SCRIPTS_ROOTDIR/training/filter-model-given-input.pl $f-$e/evaluation/filtered.$t $f-$e/tuning/moses.ini data/low/kyoto-$t.$f
        # translate
        echo "$MOSES -config $f-$e/evaluation/filtered.$t/moses.ini < data/low/kyoto-$t.$f 1> $f-$e/evaluation/$t.out 2> $f-$e/evaluation/$t.err"
        $MOSES -config $f-$e/evaluation/filtered.$t/moses.ini < data/low/kyoto-$t.$f 1> $f-$e/evaluation/$t.out 2> $f-$e/evaluation/$t.err
        # note: before evaluation of Japanese files, it may be desirable to re-tokenize the file
        #  using KyTea to match the reference more closely if you are using a different
        #  segmentation standard
        echo "$SCRIPTS_ROOTDIR/generic/multi-bleu.perl data/low/kyoto-$t.$e < $f-$e/evaluation/$t.out > $f-$e/evaluation/$t.grade"
        script/multi-bleu.perl data/low/kyoto-$t.$e < $f-$e/evaluation/$t.out > $f-$e/evaluation/$t.grade
    done
done
```

Due to that SRILM could not be succesfully compiled into MOSES, I eventually found a fix by changing the resulting moses.ini file .
Any reference of SRILM in that file was replaced by KENLM and this made the tuning process work

In testing, there was also an error. This was due to this line:
```script/multi-bleu.perl data/low/kyoto-$t.$e < $f-$e/evaluation/$t.out > $f-$e/evaluation/$t.grade```
The path of the rootdir script directory was set to script instead of $SCRIPTS_ROOTDIR/generic and therefore failed.

** This might be also why I had a higher blue score because I used this blue script instead of that in the script directory that is part of kftt folder **

After this I got the high blue score of 40 something. (I've deleted it while deleting the old data, accidentally)


