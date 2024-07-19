#!/bin/bash

usage() {
    echo "  options:"
    echo "      -b: launch behavior tree"
    echo "      -m: multi agent"
    echo "      -r: record rosbag"
    echo "      -t: launch keyboard teleoperation"
    echo "      -n: drone namespace, default is drone0"
}

# Arg parser
while getopts "bmrtn" opt; do
  case ${opt} in
    b )
      behavior_tree="true"
      ;;
    m )
      swarm="true"
      ;;
    r )
      record_rosbag="true"
      ;;
    t )
      launch_keyboard_teleop="true"
      ;;
    n )
      drone_namespace="${OPTARG}"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    : )
      if [[ ! $OPTARG =~ ^[wrt]$ ]]; then
        echo "Option -$OPTARG requires an argument" >&2
        usage
        exit 1
      fi
      ;;
  esac
done

source utils/tools.bash

# Shift optional args
shift $((OPTIND -1))

export GZ_SIM_RESOURCE_PATH=$PWD/assets/worlds:$PWD/assets/models:$GZ_SIM_RESOURCE_PATH

## DEFAULTS
behavior_tree=${behavior_tree:="false"}
swarm=${swarm:="false"}
record_rosbag=${record_rosbag:="false"}
launch_keyboard_teleop=${launch_keyboard_teleop:="false"}
drone_namespace=${drone_namespace:="drone"}

if [[ ${swarm} == "true" ]]; then
  simulation_config="sim_config/world_swarm.json"
  num_drones=3
else
  simulation_config="assets/worlds/world1.json" 
  num_drones=1
fi


# Generate the list of drone namespaces
drone_ns=()
for ((i=0; i<${num_drones}; i++)); do
  drone_ns+=("$drone_namespace$i")
done

for ns in "${drone_ns[@]}"
do
  tmuxinator start -n ${ns} -p tmuxinator/session.yml drone_namespace=${ns} simulation_config=${simulation_config} behavior_tree=${behavior_tree} &
  wait
done

if [[ ${record_rosbag} == "true" ]]; then
  tmuxinator start -n rosbag -p tmuxinator/rosbag.yml drone_namespace=$(list_to_string "${drone_ns[@]}") &
  wait
fi

if [[ ${launch_keyboard_teleop} == "true" ]]; then
  tmuxinator start -n keyboard_teleop -p tmuxinator/keyboard_teleop.yml simulation=true drone_namespace=$(list_to_string "${drone_ns[@]}") &
  wait
fi

tmuxinator start -n gazebo -p tmuxinator/gazebo.yml simulation_config=${simulation_config} &
wait

# Attach to tmux session ${drone_ns[@]}, window mission
tmux attach-session -t ${drone_ns[0]}:mission
