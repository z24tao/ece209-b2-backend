# ece209-b2-backend
Python web service for autonomous parking

## introduction
For ECE209 bake-off 2, we designed and developed an autonomous driving simulation "Takeover Parking". The goal of the project is to simulate an installable plugin to view the surroundings of a car in the form of supersonic distance sensors, then use this information to drive the vehicle. The user may drive the vehicle as normal (using keyboard control in our simulation), or switch to auto mode for the AI to take over driving. During manual mode, the AI collects data to how the user drives and trains itself with it, during auto mode, the AI drives the vehicle and evaluates each of its actions using a reward function (measuring distance and angle alignment with goal, and distance from obstacles), then trains itself using its own actions with high reward scores.

## steps
- frontend uploads data:
  - coordinates (vehicle, obstacles, goal)
  - direction (vehicle, vehicle wheels, goal)
  - training actions (acceleration / deceleration, left / right turn)
- computes distance to nearest obstacles in 8 directions
- train via random forest model
- for auto mode, backend returns predicted action

## demo video
[Demo video link](https://youtu.be/tbL9E4aYVys)

## demo figure
![figure 1](https://gyazo.com/b51919a3f13288b8dc92425068373629)
![figure 2](https://gyazo.com/aa1f76e85ee9a9737198cd4f730f8989)
![figure 3](https://gyazo.com/dd5d6a0c6a7a4bbb35d7e34cf95c5772)

## references
[car model](https://www.youtube.com/watch?v=o1XOUkYUDZU)
[scikit-learn random forest regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
