#
# To run VASP this script calls $vasp_std
# (or posibly $vasp_gam and/or $vasp_ncl).
# These variables can be defined by sourcing vaspcmd
. vaspcmd 2> /dev/null

#
# When vaspcmd is not available and $vasp_std,
# $vasp_gam, and/or $vasp_ncl are not set as environment
# variables, you can specify them here
[ -z "`echo $vasp_std`" ] && vasp_std="mpirun -np 8 /path-to-your-vasp/vasp_std"
[ -z "`echo $vasp_gam`" ] && vasp_gam="mpirun -np 8 /path-to-your-vasp/vasp_gam"
[ -z "`echo $vasp_ncl`" ] && vasp_ncl="mpirun -np 8 /path-to-your-vasp/vasp_ncl"

#
# The real work starts here
#

cp KPOINTS.6 KPOINTS

rm WAVECAR
cp INCAR.DFT INCAR
$vasp_std

cp OUTCAR OUTCAR.DFT
cp vasprun.xml vasprun.DFT.xml
cp WAVECAR WAVECAR.DFT


cp INCAR.DIAG INCAR
$vasp_std

cp OUTCAR OUTCAR.DIAG
cp vasprun.xml vasprun.DIAG.xml
cp WAVECAR WAVECAR.DIAG
cp WAVEDER WAVEDER.DIAG
# ./extract_optics.sh


cp INCAR.GW INCAR
$vasp_std

cp OUTCAR OUTCAR.G0W0
cp vasprun.xml vasprun.G0W0.xml
# ./extract_chi.sh
