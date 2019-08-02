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


echo "[$(date +"%T")]: Starting roscore"
PIDs+=($!)
roscore > /dev/null 2>&1 &
roscore_PID=$!
sleep 3
echo "[$(date +"%T")]: start Loop"
PIDs=()
WORKERNUMBER="0"
NUMBEROFWORKERS="4"
namespace="GETjag"
tf_prefix="GETjag"
world="DrlWorld5RobotsWithObjects.world"
simulator="false"
startdrlagent="false"
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
      #rosnode kill drl_gaz_robot_env_wrapper
	  kill ${PIDs[@]} >/dev/null 2>&1
	  pkill gzserver >/dev/null 2>&1
	  pkill gzclient >/dev/null 2>&1
	  pkill gazebo >/dev/null 2>&1
	  truncate -s 0 output.txt
	  kill $roscore_PID > /dev/null
	  pkill -9 roscore
	  pkill -9 rosmaster

	  sleep 0.5
      exit 0;
   fi
}

# Signale SIGINT abfangen
trap 'sighandler_INT' 2

EPISODCOUNT=0

rosparam set "/Gazebo/simulator_ready" false

i=1
while [ $i -lt 6 ] 
do  	
	rosparam set "/GETjag$i/End_of_episode" false
	rosparam set "/GETjag$i/End_of_enviroment" false 
	rosparam set "/GETjag$i/Error_in_simulator" false
	rosparam set "/GETjag$i/Ready_to_Start_DRL_Agent" false
	rosparam set "/GETjag$i/worker_ready" True
	i=$((i+1))
done

while [ $CANEXIT -lt 4 ] 
do

	truncate -s 0 output.txt
	CANEXIT=0 
	#reset watchdog
	echo -e "[$(date +"%T")]: ${GREEN}Starting gazebo with world: ${NC}$world"
	roslaunch get_gazebo_worlds getjag5.launch world:=$world gui:=$simulator > output_gazebo.txt 2>&1 &
	PIDs+=($!)
	sleep 10
	STARTTIME=$(date +%s.%N)
	rosparam set "/Gazebo/simulator_ready" true
	pkill gzclient >/dev/null 2>&1
	PIDs+=($!)
	if ($simulator -eq "true"); then		
		roslaunch  get_gazebo_worlds gzclient.launch gui:="true" > /dev/null 2>&1 &
		PIDs+=($!)
		sleep 1
	fi
		
	#loop
	while [ $CANEXIT -lt 1 ]; do
		#ros not started
		i=1
		while [ $i -lt 6 ] 
		do
			if (grep -q "Unable to set value" output_gazebo.txt); then
			
				rosparam set "/GETjag$i/Error_in_simulator" true
				rosparam set "/GETjag$i/Ready_to_Start_DRL_Agent" false

				echo -e "[$(date +"%T")] Error in simulator${NC} Unable to set value "
				i=5
				CANEXIT=3
				PIDs+=($!)
				sleep 0.5

			elif (grep -q "Segmentation fault" output_gazebo.txt); then
			
				rosparam set "/GETjag$i/Error_in_simulator" true
				rosparam set "/GETjag$i/Ready_to_Start_DRL_Agent" false

				echo -e "[$(date +"%T")] Error in simulator${NC}: Segmentation fault"
				i=5
				CANEXIT=3
				PIDs+=($!)
				sleep 0.5

			elif  ($(rosparam get /GETjag$i/End_of_enviroment)); then
							
				echo -e "[$(date +"%T")]: End reached, close the enviroment${NC}!"
				CANEXIT=4
				i=5
			
			else
				sleep 0.5
			fi
			i=$((i+1))
		done
	done
	ENDTIME=$(date +%s.%N)
	RUNTIME=$(convertsecs2hms $(echo "$ENDTIME - $STARTTIME" | bc))
	echo -e "[$(date +"%T")]: ${CYAN}Simulation on world: ${NC}$world ${CYAN}with #: ${NC}$NUMBEROFITERATIONS ${CYAN}took: ${NC}$RUNTIME seconds"
	echo "[$(date +"%T")]: Killing: ${PIDs[@]}"
	
    #rosnode kill drl_gaz_robot_env_wrapper
	kill ${PIDs[@]} >/dev/null 2>&1
	pkill gzserver >/dev/null 2>&1
	pkill gzclient >/dev/null 2>&1
	pkill gazebo >/dev/null 2>&1
	truncate -s 0 output.txt

	sleep 10
done
echo "[$(date +"%T")]: killing Ros"
kill $roscore_PID > /dev/null


