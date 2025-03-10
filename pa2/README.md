# Monte Carlo Control for Racetrack Problem

I implemented two Monte Carlo Control agents to learn how to move around a simple virtual racetrack with one right turn.

### env.py

This file contains the racetrack environment code. When you create a racetrack, you can specify a few parameters to customize it.

- width - how wide do you want the track to be overall
- height - how tall
- track-width - how wide is any point in the track
- seed - random seed for reliably generating the same tracks for testing
- move-up - how often the track moves up as opposed to right. This helps vary the track to craft more upwards moving tracks or more rightwards moving tracks

### on_policy_agent.py

Contains code to create, train, test, and display episodes from an On-Policy Monte Carlo Exploring Starts agent.

### off_policy_agent.py

Contains code to create, train, test, and display episodes from an Off-Policy Monte Carlo agent.

### test.py

Testing file to create various graphs shown the in the report.

# To Run My Code

To run my code make sure that the following packages are installed

```bash
pip install numpy matplotlib pandas imageio
```

Then run any of the methods in `test.py`
