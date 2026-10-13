#!/bin/sh
#
# Copies results of TS tests to Reference directory
#
if [ $# != 1 ] 
then
   echo "Usage: $0 Reference_Directory (no trailing /)"
   exit
fi
#
refdir=$1
#
rm -f _sources
cat > _sources <<EOF
./ts_au_100_0.25V/au_100.out
./ts_au_100_0.25V/elec_au_100.out
./ts_au_100_0.25V/tbt_au_100.out
./ts_au_100_0.25V/work/TBT_au_100/au_100.AVTRANS
./ts_au_100_0.25V/work/TBT_au_100/au_100.TEIG
./ts_au_100_0.25V/work/TBT_au_100/au_100.TRANS
./ts_au_100/au_100.out
./ts_au_100/elec_au_100.out
./ts_au_100_repetition_0.25V/au_100.out
./ts_au_100_repetition_0.25V/elec_au_100.out
./ts_au_100_repetition_0.25V/tbt_au_100.out
./ts_au_100_repetition_0.25V/work/TBT_au_100/au_100.AVTRANS
./ts_au_100_repetition_0.25V/work/TBT_au_100/au_100.TEIG
./ts_au_100_repetition_0.25V/work/TBT_au_100/au_100.TRANS
./ts_au_100_repetition/au_100.out
./ts_au_100_repetition/elec_au_100.out
./ts_au_100_repetition/tbt_au_100.out
./ts_au_100_repetition/work/TBT_au_100/au_100.AVTRANS
./ts_au_100_repetition/work/TBT_au_100/au_100.TEIG
./ts_au_100_repetition/work/TBT_au_100/au_100.TRANS
./ts_au_100/tbt_au_100.out
./ts_au_100/work/TBT_au_100/au_100.AVTRANS
./ts_au_100/work/TBT_au_100/au_100.TEIG
./ts_au_100/work/TBT_au_100/au_100.TRANS
./ts_au/au_111_capacitor.out
./ts_au/bulk_au_111.out
./ts_au/elec_au_111_abc.out
./ts_au_repetition/au_111_capacitor.out
./ts_au_repetition/elec_au_111_abc.out
./ts_au_repetition/tbt_au_111_capacitor.out
./ts_au_repetition/work/TBT_au_111_capacitor/au_111_capacitor.AVTRANS
./ts_au_repetition/work/TBT_au_111_capacitor/au_111_capacitor.TEIG
./ts_au_repetition/work/TBT_au_111_capacitor/au_111_capacitor.TRANS
./ts_au/tbt_au_111_capacitor.out
./ts_au/tbt_bulk_au_111.out
./ts_au/work/TBT_au_111_capacitor/au_111_capacitor.AVTRANS
./ts_au/work/TBT_au_111_capacitor/au_111_capacitor.TEIG
./ts_au/work/TBT_au_111_capacitor/au_111_capacitor.TRANS
./ts_au/work/TBT_bulk_au_111/bulk_au_111.AVTRANS
./ts_au/work/TBT_bulk_au_111/bulk_au_111.TEIG
./ts_au/work/TBT_bulk_au_111/bulk_au_111.TRANS
./ts_fast/ts_fast_elec.out
./ts_fast/ts_fast_scat.out
./ts_fast/ts_fast_tbt.out
./ts_fast/work/TBT/scat.fast.AVTRANS
./ts_fast/work/TBT/scat.fast.TEIG
./ts_fast/work/TBT/scat.fast.TRANS
EOF
#
for i in `cat _sources`; do
 target=`echo $i | cut -d "/" -f 2`
 echo "-- Copying $i to $target"
 cp -p $i ${refdir}/${target}
done
