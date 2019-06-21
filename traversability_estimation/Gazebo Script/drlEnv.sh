#evalscript
NUMBEROFITERATIONS=$1
declare -a worldName=("MAN2" "MOB3" "MAN1" "MAN3") #"Test_ramp" "Test_rool" "Test_stair_hard" "Test_stair_small"
declare -a robotSpeedArr=(1.0 0.8 0.6 0.4 0.2) #(1.0 0.8 0.6 0.4 0.2)
declare -a RUNDURATIONArr=(55 60 70 90 120) # (55 60 70 90 120)
declare -a maxDistance=(10.0 7.5 15.0 15.0) # (10.0 7.5 15.0 15.0)
CANEXIT=0 
echo "[$(date +"%T")]: Starting roscore"
roscore > /dev/null 2>&1 &
roscore_PID=$!
sleep 3
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
#iterate through worlds
convertsecs2hms() {
 h=$(bc <<< "${1}/3600")
 m=$(bc <<< "(${1}%3600)/60")
 s=$(bc <<< "${1}%60")
 printf "%02d:%02d:%05.2f\n" $h $m $s
}


echo "[$(date +"%T")]: start Loop"
for i in ${!worldName[@]}; do
	STARTTIME=$(date +%s.%N)
	PIDs=()
	world=${worldName[$i]}
	md=${maxDistance[$i]}
	echo -e "[$(date +"%T")]: ${GREEN}Starting gazebo with world: ${NC}$world"
	roslaunch get_gazebo_worlds getjag.launch world:=$world gui:="true" > /dev/null 2>&1 &
	PIDs+=($!)
	sleep 10
	echo "[$(date +"%T")]: Starting roscontrol"
	roslaunch get_gazebo_worlds getjag_control.launch > /dev/null 2>&1 &
	PIDs+=($!)
	sleep 3
	echo "[$(date +"%T")]: Setting Robot Random Position"
	
	rostopic pub --once /gazebo/set_model_state gazebo_msgs/ModelState '{model_name: getjag, pose: { position: { x: 1, y: 0, z: 2 }, orientation: {x: 0, y: 0.491983115673, z: 0, w: 0.870604813099 } }, twist: { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0}  }, reference_frame: world }'

	PIDs+=($!)
	
	sleep 3
	for i in ${!robotSpeedArr[@]}; do
		CANEXIT=0
		RUNDURATION=${RUNDURATIONArr[i]}
		ROBOTSPEED=${robotSpeedArr[i]}
		MAXTIME=$((RUNDURATION * 100 * NUMBEROFITERATIONS))
		#reset watchdog
		TIMEPASSED=0
		
		echo "[$(date +"%T")]: Running place_robot $ROBOTSPEED $RUNDURATION $md 0.0 0.0 0.1 $NUMBEROFITERATIONS $world"
		rosrun tutorial place_robot $ROBOTSPEED $RUNDURATION $md 0.0 0.0 0.1 $NUMBEROFITERATIONS $world > output.txt 2>&1 &
		ROSRUN_PID=($!)
		#loop until place_robot on first stage is done
		while [ $CANEXIT -lt 1 ]; do
			#watchdog timer
			if [ $((TIMEPASSED - MAXTIME)) -ge 0 ]; then
				echo -e "[$(date +"%T")]: ${RED}WATCHDOG: Timepassed: ${NC}$TIMEPASSED${RED} Maxtime: ${NC}$MAXTIME"
				CANEXIT=3
			fi
			#ros not started
			if (grep -q "Failed to call service" output.txt); then
				echo -e "[$(date +"%T")]: ${RED}ERROR: Failed to call service!${NC}"
				kill ${PIDs[@]} >/dev/null 2>&1
				kill $ROSRUN_PID >/dev/null 2>&1
				pkill gzserver >/dev/null 2>&1
				pkill gzclient >/dev/null 2>&1
				pkill gazebo >/dev/null 2>&1
				echo "[$(date +"%T")]: Killing roscore, caused by Error"
				kill $roscore_PID >/dev/null 2>&1
				sleep 10
				cwd=$(pwd)
				cd ~/.gazebo/log
				ls -t | tail -n +2 | xargs rm -rf
				sleep 10
				cd "$cwd"
			
				#restart ros
				echo "[$(date +"%T")]: Restarting roscore"
				roscore > /dev/null 2>&1 &
				roscore_PID=$!
				sleep 3
				
				#restart other
				PIDs=()
				echo -e "[$(date +"%T")]: ${GREEN}Restarting gazebo with world: ${NC}$world"
				roslaunch getjag_new getjag_world2.launch world:=$world gui:="true" > /dev/null 2>&1 &
				PIDs+=($!)
				sleep 10
				echo "[$(date +"%T")]: Restarting roscontrol"
				roslaunch getjag_new getjag_control.launch > /dev/null 2>&1 &
				PIDs+=($!)
				sleep 3
				echo "[$(date +"%T")]: Setting Flipper, arm and tower"
				rostopic pub -r 10 /GETjag/front_flipper_controller/command std_msgs/Float64 "data: -0.5" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/back_flipper_controller/command std_msgs/Float64 "data: -0.0" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_elbow_controller/command std_msgs/Float64 "data: 0.05" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_shoulder_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_socket_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_wrist_gripper_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_wrist_pitch_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_wrist_roll_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/arm_wrist_yaw_controller/command std_msgs/Float64 "data: 0.1" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/sensorhead_pitch_controller/command std_msgs/Float64 "data: 0.4" > /dev/null 2>&1 &
				PIDs+=($!)
				rostopic pub -r 10 /GETjag/sensorhead_yaw_controller/command std_msgs/Float64 "data: -0.1" > /dev/null 2>&1 &
				PIDs+=($!)
				sleep 3
				
				TIMEPASSED=0
				echo "[$(date +"%T")]: Rerun place_robot $ROBOTSPEED $RUNDURATION $md 0.0 0.0 0.1 $NUMBEROFITERATIONS $world"
				rosrun tutorial place_robot $ROBOTSPEED $RUNDURATION $md 0.0 0.0 0.1 $NUMBEROFITERATIONS $world > output.txt 2>&1 &
				ROSRUN_PID=($!)
			fi
			#execution done?
			if (grep -q "exit" output.txt); then
				echo -e "[$(date +"%T")]: Done. Switching to next speed."
				CANEXIT=2
			else
				sleep 0.5
				TIMEPASSED=$((50 + TIMEPASSED))
			fi
		done
		
		#Watchdog? HardReset...
		if [ $CANEXIT -eq 3 ]; then
			kill ${PIDs[@]} >/dev/null 2>&1
			kill $ROSRUN_PID >/dev/null 2>&1
			pkill gzserver >/dev/null 2>&1
			pkill gzclient >/dev/null 2>&1
			pkill gazebo >/dev/null 2>&1
			echo "[$(date +"%T")]: Killing roscore, caused by Watchdog"
			kill $roscore_PID >/dev/null 2>&1
			sleep 10
			cwd=$(pwd)
			cd ~/.gazebo/log
			ls -t | tail -n +2 | xargs rm -rf
			sleep 10
			cd "$cwd"
			
			#restart ros
			echo "[$(date +"%T")]: Restarting roscore"
			roscore > /dev/null 2>&1 &
			roscore_PID=$!
			sleep 3
			
			#restart other
			PIDs=()
			echo -e "[$(date +"%T")]: ${GREEN}Restarting gazebo with world: ${NC}$world"
			roslaunch getjag_new getjag_world2.launch world:=$world gui:="true" > /dev/null 2>&1 &
			PIDs+=($!)
			sleep 10
			echo "[$(date +"%T")]: Restarting roscontrol"
			roslaunch getjag_new getjag_control.launch > /dev/null 2>&1 &
			PIDs+=($!)
			sleep 3
			echo "[$(date +"%T")]: Setting Flipper, arm and tower"
			rostopic pub -r 10 /GETjag/front_flipper_controller/command std_msgs/Float64 "data: -0.5" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/back_flipper_controller/command std_msgs/Float64 "data: -0.0" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_elbow_controller/command std_msgs/Float64 "data: 0.05" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_shoulder_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_socket_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_wrist_gripper_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_wrist_pitch_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_wrist_roll_controller/command std_msgs/Float64 "data: 0.0" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/arm_wrist_yaw_controller/command std_msgs/Float64 "data: 0.1" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/sensorhead_pitch_controller/command std_msgs/Float64 "data: 0.4" > /dev/null 2>&1 &
			PIDs+=($!)
			rostopic pub -r 10 /GETjag/sensorhead_yaw_controller/command std_msgs/Float64 "data: -0.1" > /dev/null 2>&1 &
			PIDs+=($!)
			sleep 3
		fi
		sleep 1
	done
	echo "[$(date +"%T")]: Killing: ${PIDs[@]}"
	kill ${PIDs[@]} >/dev/null 2>&1
	kill $ROSRUN_PID >/dev/null 2>&1
	pkill gzserver >/dev/null 2>&1
	pkill gzclient >/dev/null 2>&1
	pkill gazebo >/dev/null 2>&1
	ENDTIME=$(date +%s.%N)
	RUNTIME=$(convertsecs2hms $(echo "$ENDTIME - $STARTTIME" | bc))
	echo -e "[$(date +"%T")]: ${CYAN}Simulation on world: ${NC}$world ${CYAN}with #: ${NC}$NUMBEROFITERATIONS ${CYAN}took: ${NC}$RUNTIME seconds"
	sleep 10
	cwd=$(pwd)
	echo "remembering: $cwd"
	cd ~/.gazebo/log
	ls -t | tail -n +2 | xargs rm -rf
	sleep 3
	cd "$cwd"
done
echo "[$(date +"%T")]: killing Ros"
kill $roscore_PID > /dev/null
