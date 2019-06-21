#!/usr/bin/env python
import subprocess
with open("outputDRL.txt", "w+") as output:
    subprocess.call(["python", "./dd_DQL_robot_navigation_fixed_memory.py"], stdout=output);