##!/bin/bash
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
ROBOTSPEED=$1
NUMBEROFITERATIONS=$2
RUNDURATION=$3

#MAXTIME=$((RUNDURATION * 100 * NUMBEROFITERATIONS * 2))
#declare -a worldName=("Test_empty")
CANEXIT=0
convertsecs2hms() {
 h=$(bc <<< "${1}/3600")
 m=$(bc <<< "(${1}%3600)/60")
 s=$(bc <<< "${1}%60")
}


echo "[$(date +"%T")]: Starting roscore"
roscore > /dev/null 2>&1 &
roscore_PID=$!
sleep 3
#iterate through worlds
echo "[$(date +"%T")]: start Loop"
PIDs=()
world="DrlTestWorld3.world"
simulator="true"
startdrlagent="true"
while [ $CANEXIT -lt 4 ] 
do

	truncate -s 0 output.txt
	CANEXIT=0

	EPISODCOUNT=0

	#reset watchdog
	echo -e "[$(date +"%T")]: ${GREEN}Starting gazebo with world: ${NC}$world"
	roslaunch get_gazebo_worlds getjag.launch world:=$world gui:=$simulator > output.txt 2>&1 &
	PIDs+=($!)
	sleep 10
	echo "[$(date +"%T")]: Starting roscontrol"
	roslaunch get_gazebo_worlds getjag_control.launch > /dev/null 2>&1 &
	PIDs+=($!)
	sleep 3
		
	while [ $CANEXIT -lt 2 ] 
	do
		truncate -s 0 output.txt

		CANEXIT=0
		echo "[$(date +"%T")]: CANEXIT $CANEXIT"

		TIMEPASSED=0

		echo "[$(date +"%T")]: Running goal Pose generator/publisher + world creator"
		roslaunch traversability_estimation create_map.launch > /dev/null 2>&1 &
		PIDs+=($!)
		sleep 1
		if ($simulator -eq "true"); then		
			pkill gzclient >/dev/null 2>&1
			PIDs+=($!)
			roslaunch get_gazebo_worlds gzclient.launch gui:="True" > /dev/null 2>&1 &
			PIDs+=($!)
			sleep 1
		fi
		STARTTIME=$(date +%s.%N)
		echo "[$(date +"%T")]: Running elevation_mapper_node"
		roslaunch elevation_mapper elevation_mapper.launch > /dev/null 2>&1 &
		PIDs+=($!)
		sleep 0.1
		if ($startdrlagent -eq "true"); then
			echo "[$(date +"%T")]: Start DRL python script"
			./dd_DQL_robot_navigation_fixed_memory.py > outputDRL.txt 2>&1 &
			PIDs+=($!)
			sleep 0.1
			startdrlagent="false"
		fi
		echo "[$(date +"%T")]: Ready to Start DRL Agent"
		echo "Ready to Start DRL Agent" >> output.txt
		#PIDs+=($!)
		sleep 0.1
		
		#loop until place_robot on first stage is done
		while [ $CANEXIT -lt 1 ]; do
			#watchdog timer
			#if [ $((TIMEPASSED - MAXTIME)) -ge 0 ]; then
			#	echo -e "[$(date +"%T")]: ${RED}WATCHDOG: Timepassed: ${NC}$TIMEPASSED${RED} Maxtime: ${NC}$MAXTIME"
			#	CANEXIT=3
			#fi
			
			#ros not started
			if (grep -q "Unable to set value" output.txt); then
				echo "Model Error restart simulator" >> output.txt
				CANEXIT=2
				PIDs+=($!)
			fi
			
			#execution done?
			if (grep -q "End of episode" output.txt); then
				echo -e "[$(date +"%T")]:End of episode reset enviroment${NC}"
				EPISODCOUNT=$((EPISODCOUNT+1))
				CANEXIT=1
				echo -e "[$(date +"%T")]:EPISODCOUNT $EPISODCOUNT${NC}"
				if [ ${EPISODCOUNT} -ge 25 ]; then
					echo -e "[$(date +"%T")]:End Episodes ${NC}"
					CANEXIT=2
				fi
			fi

			if (grep -q "End enviroment" output.txt); then
				echo -e "[$(date +"%T")]: End reached, close the enviroment${NC}!"
				CANEXIT=4
			else
				sleep 0.5
				TIMEPASSED=$((50 + TIMEPASSED))
			fi
		done
		rosnode kill /GETjag/robot_positioning_node
		rosnode kill /elevation_mapper_node


	done
	ENDTIME=$(date +%s.%N)
	RUNTIME=$(convertsecs2hms $(echo "$ENDTIME - $STARTTIME" | bc))
	echo -e "[$(date +"%T")]: ${CYAN}Simulation on world: ${NC}$world ${CYAN}with #: ${NC}$NUMBEROFITERATIONS ${CYAN}took: ${NC}$RUNTIME seconds"
	echo "[$(date +"%T")]: Killing: ${PIDs[@]}"
	kill ${PIDs[@]} >/dev/null 2>&1
	pkill gzserver >/dev/null 2>&1
	pkill gzclient >/dev/null 2>&1
	pkill gazebo >/dev/null 2>&1
	sleep 10
	#Watchdog? HardReset...
	if [ $CANEXIT -ge 3 ]; then
		kill $roscore_PID >/dev/null 2>&1
		echo "[$(date +"%T")]: Killing roscore, caused by someone"
		sleep 10
		#rm -r ~/.gazebo/log/
		echo "[$(date +"%T")]: Restarting roscore"
		roscore > /dev/null 2>&1 &
		roscore_PID=$!
		sleep 3
	fi
done
echo "[$(date +"%T")]: killing Ros"
kill $roscore_PID > /dev/null
