#!/bin/bash

SBATCH="sbatch --parsable -t 15 --reservation= "
SBATCH_TEST="$SBATCH -A project_2003959 --partition=test -t 15"
SCRIPT="run-pytorch.sh"
SCRIPT_HVD="run-pytorch-hvd.sh"

jid1a=$($SBATCH $SCRIPT pytorch_dvc_cnn_simple.py)
jid1b=$($SBATCH --dependency=afterany:$jid1a $SCRIPT pytorch_dvc_cnn_simple.py --test)

jid2a=$($SBATCH $SCRIPT pytorch_dvc_cnn_pretrained.py)
jid2b=$($SBATCH --dependency=afterany:$jid2a $SCRIPT pytorch_dvc_cnn_pretrained.py --test)

jid3a=$($SBATCH $SCRIPT pytorch_gtsrb_cnn_simple.py)
jid3b=$($SBATCH --dependency=afterany:$jid3a $SCRIPT pytorch_gtsrb_cnn_simple.py --test)

jid4a=$($SBATCH $SCRIPT pytorch_gtsrb_cnn_pretrained.py)
jid4b=$($SBATCH --dependency=afterany:$jid4a $SCRIPT pytorch_gtsrb_cnn_pretrained.py --test)

jid8a=$($SBATCH $SCRIPT_HVD pytorch_dvc_cnn_simple_hvd.py)
jid8b=$($SBATCH --dependency=afterany:$jid8a $SCRIPT pytorch_dvc_cnn_simple_hvd.py --test)

jid5=$($SBATCH $SCRIPT pytorch_20ng_cnn.py)

jid6=$($SBATCH $SCRIPT pytorch_20ng_rnn.py)

jid7=$($SBATCH $SCRIPT pytorch_20ng_bert.py)

jidx=$($SBATCH_TEST --dependency=afterany:$jid1b:$jid2b:$jid3b:$jid4b:$jid5:$jid6:$jid7:$jid8b --job-name="summary" <<EOF
#!/bin/bash
echo "** pytorch_dvc_cnn ($jid1a,$jid2a -> $jid1b,$jid2b) **"
grep -h -B 1 'Accuracy' --no-group-separator slurm-{$jid1b,$jid2b}.out
echo
echo "** pytorch_dvc_cnn_hvd ($jid8a -> $jid8b) **"
grep -h -B 1 'Accuracy' --no-group-separator slurm-{$jid8b}.out
echo
echo "** pytorch_gtsrb_cnn ($jid3a,$jid4a -> $jid3b,$jid4b) **"
grep -h -B 1 'Accuracy' --no-group-separator slurm-{$jid3b,$jid4b}.out
echo
echo "** pytorch_20ng_cnn ($jid5) **"
grep -A 1 'Test set' slurm-${jid5}.out
echo
echo "** pytorch_20ng_rnn ($jid6)**"
grep -A 1 'Test set' slurm-${jid6}.out
echo
echo "** pytorch_20ng_bert ($jid7)**"
grep -A 1 'Test set' slurm-${jid7}.out
EOF
)

squeue -u $USER -p test,gpu -o "%.10i %.9P %.16j %.8T %.10M %.50E"

echo
echo "Final summary will appear in slurm-${jidx}.out"
