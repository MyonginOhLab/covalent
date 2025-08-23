### Written by Myongin Oh
### Last modification: Aug 23, 2025

mol new input.psf
mol addfile prod.dcd first 0 last -1 step 1 waitfor -1

set file_out output_coords_res.csv
set outfile [open $file_out w]

set nf [molinfo top get numframes]
set residList0 [[atomselect top "protein and name CA and (resid 5 to 132)"] get resid]
set residList [lsort -unique -integer -increasing $residList0]

set minResid [tcl::mathfunc::min {*}$residList]
set maxResid [tcl::mathfunc::max {*}$residList]
set minIndex [lsearch $residList $minResid]
set maxIndex [lsearch $residList $maxResid]

puts "minIdex = $minIndex , maxIndex = $maxIndex , Length = [llength $residList]"

set xyzList {x y z}

for {set m $minIndex} {$m <= $maxIndex} {incr m} {
	set n [lindex $residList $m]
	foreach elem $xyzList {
		puts -nonewline $outfile "res$n.$elem,"
	}
	unset n
}
puts -nonewline $outfile "\n"

for {set i 0} {$i < $nf} {incr i} {
	set distList {}
	animate goto $i
	for {set p $minIndex} {$p <= $maxIndex} {incr p} {
		set pGly [atomselect top "protein and resid [lindex $residList $p] and name CA"]
		if {[$pGly get resname] == "GLY"} {
			set sel1 [atomselect top "protein and resid [lindex $residList $p] and name HA2"]
			$sel1 frame $i
			$sel1 update
		} else {
			set sel1 [atomselect top "protein and resid [lindex $residList $p] and not (hydrogen or name O C CA N)"]
			$sel1 frame $i
			$sel1 update
		}
		set xCoord [lindex [measure center $sel1] 0]
		lappend distList $xCoord
		set yCoord [lindex [measure center $sel1] 1]
		lappend distList $yCoord
		set zCoord [lindex [measure center $sel1] 2]
		lappend distList $zCoord

		$sel1 delete
		$pGly delete
		unset xCoord
		unset yCoord
		unset zCoord
	}
	foreach elem $distList {
		puts -nonewline $outfile "$elem,"
	}
	unset -nocomplain distList
	puts -nonewline $outfile "\n"
}

unset nf
unset -nocomplain residList0
unset -nocomplain residList
unset -nocomplain xyzList
unset minResid
unset maxResid
unset minIndex
unset maxIndex

close $outfile