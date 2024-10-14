FROM osrf/ros:humble-desktop

# minimal-sensing-exploration

WORKDIR /root/

RUN apt-get update && \
    apt-get install apt-utils -y \
        software-properties-common \
        git tmux tmuxinator \
        python3-rosdep \
        python3-pip \
        python3-colcon-common-extensions \
        python3-colcon-mixin \
        python-is-python3 \
        ros-dev-tools \
        python3-flake8 \
        python3-flake8-builtins  \
        python3-flake8-comprehensions \
        python3-flake8-docstrings \
        python3-flake8-import-order \
        python3-flake8-quotes \
        cppcheck lcov \
        lsb-release wget gnupg \
        ros-humble-grid-map-rviz-plugin \
    && rm -rf /var/lib/apt/lists/* 

RUN pip3 install \
        pylint \
        flake8==4.0.1 \
        pycodestyle==2.8 \
        cmakelint \
        cpplint \
        colcon-lcov-result \
        PySimpleGUI-4-foss

RUN colcon mixin update default
RUN rm -rf log # remove log folder

RUN mkdir -p /root/aerostack2_ws/src/
WORKDIR /root/aerostack2_ws/src/
RUN git clone https://github.com/pariaspe/aerostack2-mininal-sensing-exploration.git
RUN touch aerostack2-mininal-sensing-exploration/as2_hardware_drivers/COLCON_IGNORE
RUN git clone https://github.com/aerostack2/aerostack2_viz.git
RUN touch aerostack2_viz/as2_rviz_plugins/COLCON_IGNORE

WORKDIR /root/aerostack2_ws
RUN rosdep update
RUN apt update && rosdep install --from-paths src --ignore-src -r -y

# Install Gazebo Harmonic
# RUN apt-get update \
#     && wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
#     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null \
#     && apt-get update && apt-get install -y -q \
#         gz-harmonic \
#     && rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/$ROS_DISTRO/setup.sh && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
RUN echo 'export AEROSTACK2_PATH=/root/aerostack2_ws/src/aerostack2' >> ~/.bashrc
RUN echo 'source $AEROSTACK2_PATH/as2_cli/setup_env.bash' >> ~/.bashrc
RUN echo "source /root/aerostack2_ws/install/setup.bash" >> ~/.bashrc

WORKDIR /
RUN git clone https://github.com/aerostack2/project_gz_minimal_sensing_exploration.git gz_exploration
WORKDIR /gz_exploration

CMD ["bash"]