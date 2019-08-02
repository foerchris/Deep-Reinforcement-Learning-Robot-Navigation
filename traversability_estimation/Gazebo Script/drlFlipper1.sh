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
	while [ $i -lt 2 ] 
	do
		roslaunch elevation_mapper elevation_mapper.launch namespace:="$namespace$i" > /dev/null 2>&1 &		
		PIDs+=($!)
		i=$((i+1))
	done
	sleep 1

	echo "[$(date +"%T")]: Running goal Pose generator/publisher + world creator"
	i=1
	while [ $i -lt 2 ]
	do
		roslaunch traversability_estimation gazebo_control.launch namespace:="$namespace$i" > /dev/null 2>&1 &
		PIDs+=($!)
		i=$((i+1))
	done
	sleep 2

	echo "[$(date +"%T")]: Drive to goal Pose node"
	i=1
	while [ $i -lt 2 ] 
	do  	
		roslaunch robot_navigation auto_robot_navigation.launch namespace:="$namespace$i" > /dev/null 2>&1 &
		PIDs+=($!)
		i=$((i+1))
	done
	sleep 2
		
	i=1
	while [ $i -lt 2 ]
	do
		#roslaunch flipper_control FlipperControl.launch namespace:="$namespace$i" > /dev/null 2>&1 &
		PIDs+=($!)
		i=$((i+1))
	done
	
	echo "[$(date +"%T")]: Ready to Start DRL Agent"
	i=1
	while [ $i -lt 2 ] 
	do
		rosparam set "/GETjag$i/Ready_to_Start_DRL_Agent" true
		i=$((i+1))
	done

	sleep 1
		
	rosnode list >> cheakNotes.txt 
	sleep 1

	i=1
	while [ $i -lt 2 ] 
	do
		if (grep -q "/GETjag$i/auto_robot_navigation_server" cheakNotes.txt and grep -q "/GETjag$i/elevation_mapper_node" cheakNotes.txt and grep -q "/GETjag$i/gazebo_control_node" cheakNotes.txt); then
			CANEXIT=0
			if ($drlagentStartet -eq "true"); then		
				if (grep -q "GETjag_drl_gaz_robot_env_wrapper" cheakNotes.txt); then
					CANEXIT=0
				else
					echo -e "[$(date +"%T")]:Error GETjag_drl_gaz_robot_env_wrapper crash somehow will end simulation${NC}"
					rosparam set "/GETjag$i/Ready_to_Start_DRL_Agent" false
					CANEXIT=4
				fi
			fi		
		else
			echo -e "[$(date +"%T")]  Nodes not startet: GETjag$i/robot_positioning_node and /GETjag$i/elevation_mapper_node{NC}"

			CANEXIT=1
		fi	
		i=$((i+1))
	done
		
		
	#loop
	while [ $CANEXIT -lt 1 ]; do
		#ros not started
		i=1
		while [ $i -lt 2 ] 
		do
			if  ($(rosparam get /GETjag$i/End_of_enviroment)); then
							
				echo -e "[$(date +"%T")]: End reached, close the enviroment${NC}!"
				CANEXIT=4
				i=5
			
			else
				#/scratch-local/cdtemp/chfo/ros/packages/ros_traversability/traversability_estimation-master/traversability_estimation/mazegenerator-master/src/mazegen -m 0 -w 4 -h 4 -a 2 -t 0 -n $i > /dev/null 2>&1 &
				#/home/chfo/getbot/ros/robocup/traversability_estimation-master/traversability_estimation/mazegenerator-master/src/mazegen -m 0 -w 4 -h 4 -a 2 -t 0 -n $i > /dev/null 2>&1 &
				sleep 1
			fi
			i=$((i+1))
		done
	done
	i=1
	while [ $i -lt 2 ] 
	do
		rosnode kill "/GETjag$i/robot_positioning_node"
		rosnode kill "/GETjag$i/elevation_mapper_node"
		i=$((i+1))
	done
done
