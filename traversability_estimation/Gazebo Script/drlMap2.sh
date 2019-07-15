##!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color



#declare -a worldName=("Test_empty")
CANEXIT=0
convertsecs2hms() {
 h=$(bc <<< "${1}/3600")
 m=$(bc <<< "(${1}%3600)/60")
 s=$(bc <<< "${1}%60")
}



PIDs=()
WORKERNUMBER="0"
NUMBEROFWORKERS="4"
namespace="GETjag"
tf_prefix="GETjag"
drlagentStartet="false"

# Demonstriert die Funktion trap zum Abfangen von Signalen
# Name: trap4
sighandler_INT() {
   printf "Habe das Signal SIGINT empfangen\n"
   printf "Soll das Script beendet werden (j/n) : "
   read
   if [[ $REPLY = "j" ]]
   then
      echo "Bye!"
	  rosnode kill "/GETjag$i/robot_positioning_node"
	  rosnode kill "/GETjag$i/elevation_mapper_node"
	  rosnode kill "/GETjag$i/flipper_control_server"
	  kill ${PIDs[@]} >/dev/null 2>&1

	  sleep 0.5
      exit 0;
   fi
}

# Signale SIGINT abfangen
trap 'sighandler_INT' 2

EPISODCOUNT=0

PIDs+=($!)

while [ $CANEXIT -lt 2 ] 
do
	truncate -s 0 cheakNotes.txt
		
	echo "[$(date +"%T")]: Running elevation_mapper_node"
	i=1
	while [ $i -lt 3 ]
	do
		roslaunch elevation_mapper elevation_mapper.launch namespace:="$namespace$i" > /dev/null 2>&1 &		
		PIDs+=($!)
		i=$((i+1))
	done
	sleep 1

		
	#loop
	while [ $CANEXIT -lt 1 ]; do
		#ros not started
		i=1
		while [ $i -lt 3 ]
		do
			if  ($(rosparam get /GETjag$i/End_of_enviroment)); then
							
				echo -e "[$(date +"%T")]: End reached, close the enviroment${NC}!"
				CANEXIT=4
				i=5
			
			else
				sleep 1
			fi
			i=$((i+1))
		done
	done
	i=1
	while [ $i -lt 3 ]
	do
		rosnode kill "/GETjag$i/robot_positioning_node"
		rosnode kill "/GETjag$i/elevation_mapper_node"
		i=$((i+1))
	done
done
