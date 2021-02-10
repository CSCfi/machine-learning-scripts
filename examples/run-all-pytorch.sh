#!/bin/bash

SBATCH="sbatch --parsable -t 15 --reservation= "
SBATCH_TEST="$SBATCH -A project_2003959 --partition=test -t 15"
SCRIPT="run-pytorch.sh"
SCRIPT_HVD="run-pytorch-hvd.sh"

jid1=$($SBATCH $SCRIPT pytorch_dvc_cnn_simple.py)

jid2=$($SBATCH $SCRIPT pytorch_dvc_cnn_pretrained.py)

jid3=$($SBATCH $SCRIPT pytorch_gtsrb_cnn_simple.py)

jid4=$($SBATCH $SCRIPT pytorch_gtsrb_cnn_pretrained.py)

jid8=$($SBATCH $SCRIPT_HVD pytorch_dvc_cnn_simple_hvd.py)

jid5=$($SBATCH $SCRIPT pytorch_20ng_cnn.py)

jid6=$($SBATCH $SCRIPT pytorch_20ng_rnn.py)

jid7=$($SBATCH $SCRIPT pytorch_20ng_bert.py)

jidx=$($SBATCH_TEST --dependency=afterany:$jid1:$jid2:$jid3:$jid4:$jid5:$jid6:$jid7:$jid8 --job-name="summary" <<EOF
#!/bin/bash
echo "** pytorch_dvc_cnn ($jid1,$jid2) **"
grep -h -A 1 '^Simple:' slurm-${jid1}.out
grep -h -A 1 -E '^Pretrained:|^Finetuned:' --no-group-separator slurm-${jid1}.out
echo
echo "** pytorch_dvc_cnn_hvd ($jid8) **"
grep -h -B 1 'Accuracy' --no-group-separator slurm-${jid8}.out
echo
echo "** pytorch_gtsrb_cnn ($jid3,$jid4) **"
grep -h -A 1 '^Simple:' slurm-${jid3}.out
grep -h -A 1 -E '^Pretrained:|^Finetuned:' --no-group-separator slurm-${jid4}.out
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
