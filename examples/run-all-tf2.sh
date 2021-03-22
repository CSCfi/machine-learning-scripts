#!/bin/bash

SBATCH="sbatch --parsable -t 15 --reservation= "
SBATCH_TEST="$SBATCH -A project_2003959 --partition=test -t 15"
SCRIPT="run.sh"
SCRIPT_HVD="run-hvd.sh"

jid1a=$($SBATCH $SCRIPT tf2-dvc-cnn-simple.py)
jid1b=$($SBATCH --dependency=afterany:$jid1a $SCRIPT tf2-dvc-cnn-evaluate.py dvc-cnn-simple.h5)

jid2a=$($SBATCH $SCRIPT tf2-dvc-cnn-pretrained.py)
jid2b=$($SBATCH --dependency=afterany:$jid2a $SCRIPT tf2-dvc-cnn-evaluate.py dvc-vgg16-reuse.h5)
jid2c=$($SBATCH --dependency=afterany:$jid2a $SCRIPT tf2-dvc-cnn-evaluate.py dvc-vgg16-finetune.h5)

jid7a=$($SBATCH $SCRIPT tf2-dvc_tfr-cnn-simple.py)
jid7b=$($SBATCH --dependency=afterany:$jid7a $SCRIPT tf2-dvc_tfr-cnn-evaluate.py dvc_tfr-cnn-simple.h5)

jid8a=$($SBATCH $SCRIPT tf2-dvc_tfr-cnn-pretrained.py)
jid8b=$($SBATCH --dependency=afterany:$jid8a $SCRIPT tf2-dvc_tfr-cnn-evaluate.py dvc_tfr-vgg16-reuse.h5)
jid8c=$($SBATCH --dependency=afterany:$jid8a $SCRIPT tf2-dvc_tfr-cnn-evaluate.py dvc_tfr-vgg16-finetune.h5)

jid9a=$($SBATCH $SCRIPT_HVD tf2-dvc-cnn-simple-hvd.py)
jid9b=$($SBATCH --dependency=afterany:$jid9a $SCRIPT tf2-dvc-cnn-evaluate.py dvc-cnn-simple-hvd.h5)

jid3a=$($SBATCH $SCRIPT tf2-gtsrb-cnn-simple.py)
jid3b=$($SBATCH --dependency=afterany:$jid3a $SCRIPT tf2-gtsrb-cnn-evaluate.py gtsrb-cnn-simple.h5)

jid4a=$($SBATCH $SCRIPT tf2-gtsrb-cnn-pretrained.py)
jid4b=$($SBATCH --dependency=afterany:$jid4a $SCRIPT tf2-gtsrb-cnn-evaluate.py gtsrb-vgg16-reuse.h5)
jid4c=$($SBATCH --dependency=afterany:$jid4a $SCRIPT tf2-gtsrb-cnn-evaluate.py gtsrb-vgg16-finetune.h5)

jid5=$($SBATCH $SCRIPT tf2-20ng-cnn.py)

jid6=$($SBATCH $SCRIPT tf2-20ng-rnn.py)

jid10=$($SBATCH $SCRIPT tf2-20ng-bert.py)

jidx=$($SBATCH_TEST --dependency=afterany:$jid1b:$jid2b:$jid2c:$jid3b:$jid4b:$jid4c:$jid5:$jid6:$jid7b:$jid8b:$jid8c:$jid9b:$jid10 --job-name="summary" <<EOF
#!/bin/bash
echo "** tf2-dvc-cnn ($jid1a,$jid2a -> $jid1b,$jid2b,$jid2c) **"
grep -h --no-group-separator -E 'Evaluating|Test set accuracy' slurm-{$jid1b,$jid2b,$jid2c}.out
echo
echo "** tf2-dvc_tfr-cnn ($jid7a,$jid8a -> $jid7b,$jid8b,$jid8c) **"
grep -h --no-group-separator -E 'Evaluating|Test set accuracy' slurm-{$jid7b,$jid8b,$jid8c}.out
echo
echo "** tf2-dvc-cnn-simple-hvd ($jid9a -> $jid9b) **"
grep -h --no-group-separator -E 'Evaluating|Test set accuracy' slurm-${jid9b}.out
echo
echo "** tf2-gtsrb-cnn ($jid3a, $jid4a -> $jid3b,$jid4b,$jid4c)**"
grep -h --no-group-separator -E 'Evaluating|Test set accuracy' slurm-{$jid3b,$jid4b,$jid4c}.out
echo
echo "** tf2-20ng-cnn ($jid5) **"
grep 'Test set accuracy' slurm-${jid5}.out
echo
echo "** tf2-20ng-rnn ($jid6)**"
grep 'Test set accuracy' slurm-${jid6}.out
echo
echo "** tf2-20ng-bert ($jid10)**"
grep 'Test set accuracy' slurm-${jid10}.out
EOF
)

squeue -u $USER -o "%.10i %.9P %.16j %.8T %.10M %.50E" -p gpu,test

echo
echo "Final summary will appear in slurm-${jidx}.out"
