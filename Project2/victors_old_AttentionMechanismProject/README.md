# AttentionMechanismProject
An Honours project by Cornelis Boon and Victor Milewski, supervised by Ke Tran and Katya Garmash, about Attention Mechanisms. 


# Pre-ordering for SMT
We propose neural attention models to address pre-ordering in an elegant way

## Reference
[Reordering Grammar Induction](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP005.pdf)

Read citation for further references

## Data processing

See [Moses tutorial](http://www.statmt.org/moses/?n=Moses.Tutorial)

## Building the baseline (without pre-ordering)

Create a experiment directory

```
$ mkdir experiments
```

In `experiments` put all your data

`experiments\data\tok`

Move process.sh script into `experiments`
Edit `process.sh` with your favorite editor, you need to change these lines

```
# Replace the following path by your own path
MT_SCRIPTS=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/scripts/tokenizer
SCRIPTS_ROOTDIR=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/scripts
MOSES_DIR=/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder
```

Also make sure that Moses and SRILM is visible in your environment. Add following lines to your bash file

```
# .bash_profile
# point to Moses and local sirlm
export PATH=$PATH:/zfs/ilps-plexest/smt/ke/mysoftware/srilm-1.7.1/bin/i686-m64
export PATH=$PATH:/zfs/ilps-plexest/smt/ke/mysoftware/mosesdecoder/bin

```

then update the environment

```
source ~/.bash_profile
```

Now you are ready to obtain the baseline

```
$ nohup ./process.sh &> process.log &
```

Note: it's better to use `mgiza` than `giza++`, to do so
you need to copy binaries of mgiza to `mosesdecoder/tools`
then pass appropriate parameters to train-model.perl (see http://www.statmt.org/moses/?n=Moses.Optimize#ntoc9)

