#!/bin/bash

# TODO
# pytorch_dvc_cnn_hvd.py
# pytorch_dvc_cnn_simple_hvd.py

SBATCH="sbatch --parsable"
SCRIPT="run-puhti-pytorch.sh"

jid1a=$($SBATCH $SCRIPT pytorch_dvc_cnn_simple.py)
jid1b=$($SBATCH --dependency=afterany:$jid1a $SCRIPT pytorch_dvc_cnn_simple.py --test)

jid2a=$($SBATCH $SCRIPT pytorch_dvc_cnn_pretrained.py)
jid2b=$($SBATCH --dependency=afterany:$jid2a $SCRIPT pytorch_dvc_cnn_pretrained.py --test)

jid3a=$($SBATCH $SCRIPT pytorch_gtsrb_cnn_simple.py)
jid3b=$($SBATCH --dependency=afterany:$jid3a $SCRIPT pytorch_gtsrb_cnn_simple.py --test)

jid4a=$($SBATCH $SCRIPT pytorch_gtsrb_cnn_pretrained.py)
jid4b=$($SBATCH --dependency=afterany:$jid4a $SCRIPT pytorch_gtsrb_cnn_pretrained.py --test)

jid5=$($SBATCH $SCRIPT pytorch_20ng_cnn.py)
jid6=$($SBATCH $SCRIPT pytorch_20ng_rnn.py)

jidx=$($SBATCH -A project_2001756 --partition=test --dependency=afterany:$jid1b:$jid2b:$jid3b:$jid4b:$jid5:$jid6 --job-name="summary" <<EOF
#!/bin/bash
echo "** pytorch_dvc_cnn **"
grep -h -B 1 'Accuracy' --no-group-separator slurm-{$jid1b,$jid2b}.out
echo
echo "** pytorch_dvc_gtsrb **"
grep -h -B 1 'Accuracy' --no-group-separator slurm-{$jid3b,$jid4b}.out
EOF
)

squeue -u $USER -o "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R %.20E"

echo
echo "Final summary will appear in slurm-${jidx}.out"
