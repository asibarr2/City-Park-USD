# City-Park-USD
USD File for city park. Used for Reinforcement Learning training


Ensure that all the assests are located in one folder. This will allow the scene to generate without errors.

# (1) City Park Environment Setup for Reinforcement Learning

The __CityView.zip__ folder contains all the assests for the city. Extract the folder and upload it to Omniverse Nucleus in localhost -> Isaac -> Environments. You must place it in this directory since this is where the code is referenced to for the RL training. If you want to change the directory, you will have to go to into the __stable\_baselines__ folder and edit __testenv.py__ on line 60 to change the path of the USD file. The line should look like this:

  self.usd_path = nucleus_server + "/Isaac/Environments/CityView/Props/S_Park_Grounds.usd"

This directory points to a prop called S_Park__Ground.usd, which 

### (2) Carter Libary Setup

In the ~/.local/ov/share/pkg/isaac_sim-20xx.x.x/exts/omni.isaac.jetbot/omni/isaac/jetbot directory (or wherever  your isaac sim configuration is located), paste the following file from __(1)Carter Py File__  on the USB into the current jetbot directory as indicated before:

carter.py

This file setups the carter USD file to be used in the RL training system. In order to ensure this package is identified in the Isaac Sim package, replace the \_\_init\_\_.py with the new \_\_init.\_\_.py located in __Carter Py File__ directory. This should setup the carter libary to import the Carter USD into  Isaac Sim for RL Training



### (3) Reinforcement Learning Setup

Copy and paste all files from __stable\_baselines\_example__  into ~/.local/ov/share/pkg/isaac_sim-20xx.x.x/standalone_examples/api/omni.isaac.jetbot/stable_baselines_example. 

To train the system, go into ~/.local/ov/share/pkg/isaac_sim-20xx.x.x/ and type the following to train the system:

./python.sh standalone_examples/api/omni.isaac.jetbot/stable_baselines_example/trainPPO.py



# Contact Information if there are errors

Email: asibarra98@gmail.com

Alternate Email: alexander.ibarra516@gmail.com

Mobile: 210-275-9496



